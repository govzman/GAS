import contextlib
import lpips

import torch
from torch import nn
from typing import Tuple, Optional, Any, List, Dict
from torch.nn.functional import interpolate
from torch_ema import ExponentialMovingAverage
from ml_collections import ConfigDict
from hydra.utils import instantiate

from src.model.gas.base_model import BaseModel
from src.model.gas.generalized_solver import GeneralizedSolver
from src.model.gas.adversarial_module.dist_adv_loss import DistAdversarialTraining
from src.model.gas.synt_data import SyntDataType
from src.model.gas.stepwise_coeff_predictor import StepwiseCoeffPredictor
from src.scheduler_model.film_mlp import PromptNoiseFiLMMlp


def _final_linear_scheduler_transformer(module: nn.Module) -> Optional[nn.Linear]:
    """Последний `Linear` в голове `SchedulerTransformer`: `*.mlps.mlps[-1]` (вложенный MLP)."""
    inner = getattr(module, "mlps", None)
    if inner is None or not hasattr(inner, "mlps"):
        return None
    seq = inner.mlps
    if not isinstance(seq, nn.Sequential) or len(seq) == 0:
        return None
    tail = seq[-1]
    return tail if isinstance(tail, nn.Linear) else None


def _init_scheduler_transformer_residual_head(
    module: nn.Module,
    logistic_bias_1d: Optional[torch.Tensor] = None,
) -> None:
    """
    Вес последнего линейного слоя в нуль (остаток к теоретическим a/c или к нулевому сдвигу t);
    bias — нули или заданный вектор логитов (inv stick-breaking для равномерной сетки t), если совпадает размерность.
    """
    lin = _final_linear_scheduler_transformer(module)
    if lin is None:
        return
    nn.init.zeros_(lin.weight)
    if lin.bias is None:
        return
    if logistic_bias_1d is not None and logistic_bias_1d.numel() == lin.bias.numel():
        with torch.no_grad():
            lin.bias.copy_(logistic_bias_1d.to(device=lin.bias.device, dtype=lin.bias.dtype))
    else:
        nn.init.zeros_(lin.bias)
    

class GSWrapper(nn.Module):
    """Generalised Solver wrapper. 
    
    This class integrates all the logic needed to train or evaluate 
    the given generative model using Generalised Solver. 
    
    Method `student_sampler_fn` is used to call the sampler.
    Method `forward` is called in the training loop to calculate the losses.
        The model can be trained in both default or adversarial modes.
    
    Attributes:
        model (BaseModel): Underlying model instance wrapped in BaseModel interface.
        solver_config (ConfigDict): Solver configuration dictionary.
        solver (GeneralizedSolver): Generalised Solver instance that is trained/evaluated.
        
        loss_fn_vgg (nn.Module): VGG model instance to calculate LPIPS loss.
        adv_loss (DistAdversarialTraining): Adversarial training class instance. 
    """
    
    def __init__(self, model: BaseModel, config: ConfigDict):
        """Initialize the Generalised Solver wrapper.
    
        Args:
            model (BaseModel): Instance of a BaseModel class. 
                Its `decode`, `set_condition` methods and 
                `model_fn`, `ns` and `t_eps` attributes are used.
            config (ConfigDict): config, also include solver configuration dictionary.
                Must include steps, order, loss_config, 
                t_parametrization and use_theory_coef.
        """
        super().__init__()
        self.model = model
        self.solver_config = config.student_solver
        self.t_eps = self.model.t_eps
        
        self.shuffle_coef_noise = getattr(self.solver_config, "shuffle_coef_noise", False)
        self.shuffle_coef_conditioning = getattr(self.solver_config, "shuffle_coef_conditioning", False)

        # Backward compatibility
        if getattr(self.solver_config, "shuffle_conditioning", False):
            self.shuffle_coef_noise = True
            self.shuffle_coef_conditioning = True

        if self.shuffle_coef_noise or self.shuffle_coef_conditioning:
            print(f"⚠️ WARNING: Coefficient shuffle enabled - noise: {self.shuffle_coef_noise}, conditioning: {self.shuffle_coef_conditioning}")
        
        # create lpips
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').requires_grad_(False)
        self.loss_fn_vgg.eval()

        # construct loss
        self.loss_config = config.loss_config

        # REPA (Representation Alignment, frozen DINOv2)
        self.use_repa = getattr(self.loss_config, "use_repa", False)
        self.use_repa_in_eval = getattr(self.loss_config, "use_repa_in_eval", False)
        self.repa_loss = None
        if self.use_repa:
            from src.model.gas.repa_loss import FinalREPALoss

            self.repa_loss = FinalREPALoss(
                model_name=getattr(self.loss_config, "repa_model", "dinov2_vitb14"),
                layer_indices=getattr(self.loss_config, "repa_layer_indices", None),
                loss_type=getattr(self.loss_config, "repa_loss_type", "cosine"),
                layer_weights=getattr(self.loss_config, "repa_layer_weights", None),
                use_cls_token=getattr(self.loss_config, "repa_use_cls_token", True),
                use_patch_tokens=getattr(self.loss_config, "repa_use_patch_tokens", True),
                normalize_features=getattr(self.loss_config, "repa_normalize_features", True),
                pretrained=getattr(self.loss_config, "repa_pretrained", True),
            )
            self.repa_loss.eval()

        assert self.loss_config.loss_type in ["GS", "GAS"]
        if self.loss_config.loss_type == "GAS":
            self.adv_loss = DistAdversarialTraining(self.loss_config)
        
        self.freeze_mlp_steps = int(getattr(self.solver_config, 'freeze_mlp_steps', 0))
        self.mlp_disabled = (self.freeze_mlp_steps > 0)  # ⭐ Новый флаг вместо mlp_frozen

        if self.freeze_mlp_steps > 0:
            print(f"🧊 MLP will be DISABLED (not frozen) for first {self.freeze_mlp_steps} steps")

        self.use_stepwise_coeff = bool(getattr(self.solver_config, "use_stepwise_coeff", False))
        self.stepwise_predictor: Optional[StepwiseCoeffPredictor] = None
        self._stepwise_coeff_buffers: Optional[Dict[str, torch.Tensor]] = None
        self._stepwise_sampling_timesteps: Optional[torch.Tensor] = None

        # setup solver
        solver = self.get_base_solver()
        self.steps = self.solver_config.steps
        self.order = self.solver_config.order

        # init t steps
        self.eps_mu_offset = 1e-5
        # support both new and legacy config key names
        self.t_parametrization = getattr(
            self.solver_config,
            "t_parametrization",
            getattr(self.solver_config, "t_schedule_parametrization", "diff"),
        )

        if self.t_parametrization in ("diff", "mu_logit"):
            self.mu_logit = nn.Parameter(torch.ones(self.steps - 1), requires_grad=self.solver_config.t_requires_grad)
            t_unif = torch.linspace(1., self.t_eps, self.steps + 1).flip(0)
            self.mu_logit.data = self.get_inv_t_steps(t_unif)
        elif self.t_parametrization in ("film_mlp", "prompt_noise_film_mlp"):
            hidden_dim = int(getattr(self.solver_config, "t_film_hidden_dim", 256))
            self.mu_logit = PromptNoiseFiLMMlp(out_dim=self.steps - 1, hidden_dim=hidden_dim)
        elif self.t_parametrization == "transformer":
            self.mu_logit = instantiate(config.scheduler_model)
        else:
            raise ValueError(f"Unsupported t_parametrization={self.t_parametrization}")

        solver.get_time_steps = lambda *args, **kwargs: self.get_t_steps(*args, **kwargs)

        # shared dimensions for conditional predictors
        self.latent_channels = int(getattr(self.solver_config, "latent_channels", 4))
        self.text_embed_dim = int(getattr(self.solver_config, "text_embed_dim", 768))

        self.use_shared_ac_backbone = getattr(self.solver_config, "use_shared_ac_backbone", False)
        if self.use_stepwise_coeff:
            stepwise_cfg = getattr(self.solver_config, "stepwise_predictor", None)
            if stepwise_cfg is not None:
                stepwise_cfg = dict(stepwise_cfg)
                stepwise_cfg.setdefault("order", self.order)
                stepwise_cfg.setdefault("latent_channels", self.latent_channels)
                stepwise_cfg.setdefault("text_embed_dim", self.text_embed_dim)
                self.stepwise_predictor = instantiate(stepwise_cfg)
            else:
                self.stepwise_predictor = StepwiseCoeffPredictor(
                    order=self.order,
                    latent_channels=self.latent_channels,
                    text_embed_dim=self.text_embed_dim,
                    hidden_dim=int(getattr(self.solver_config, "stepwise_hidden_dim", 256)),
                )
            print("🔁 Stepwise coefficient prediction enabled")
            if self.use_shared_ac_backbone:
                print(
                    "⚠️ use_stepwise_coeff=True: shared_ac_backbone is not used during sampling; "
                    "stepwise_predictor drives per-step coefficients."
                )
        
        if self.use_shared_ac_backbone:
            # 1. Создаём noise encoder из конфига
            noise_enc_cfg = self.solver_config.noise_encoder
            if noise_enc_cfg is None:
                raise ValueError("use_shared_ac_backbone=True, но не задан noise_encoder")
            # Передаём in_channels, если конфиг это поддерживает
            noise_enc_cfg = dict(noise_enc_cfg)  # копия, чтобы не мутировать исходный
            if 'in_channels' not in noise_enc_cfg:
                noise_enc_cfg['in_channels'] = self.latent_channels
            noise_encoder = instantiate(noise_enc_cfg)

            # 2. Создаём сам ac backbone (SharedPromptNoiseFiLMBackbone с головами)
            ac_backbone_cfg = self.solver_config.ac_backbone
            if ac_backbone_cfg is None:
                raise ValueError("use_shared_ac_backbone=True, но не задан ac_backbone")
            ac_backbone_cfg = dict(ac_backbone_cfg)
            # Передаём размерности выходов a и c (order*steps)
            ac_backbone_cfg['a_out_dim'] = self.order * self.steps
            ac_backbone_cfg['c_out_dim'] = self.order * self.steps
            # Передаём уже созданный noise_encoder, чтобы не создавать второй раз
            ac_backbone_cfg['noise_encoder'] = noise_encoder
            # prompt_dim и др. можно не передавать, если они зафиксированы в конфиге
            self.shared_ac_backbone = instantiate(ac_backbone_cfg)

            # Старые модели больше не нужны
            self.a_diff_model = None
            self.c_diff_model = None

        # init t_couple
        self.t_couple_parametrization = getattr(self.solver_config, "t_couple_parametrization", "diff")
        self.t_couple_model = None
        self._t_couple_bias = None

        if self.t_couple_parametrization == "diff":
            self.t_couple = nn.Parameter(torch.zeros(self.steps), requires_grad=self.solver_config.t_couple_requires_grad)
        else:
            if self.t_couple_parametrization in ("film_mlp", "prompt_noise_film_mlp"):
                hidden_dim = int(getattr(self.solver_config, "t_couple_film_hidden_dim", 256))
                self.t_couple_model = PromptNoiseFiLMMlp(out_dim=self.steps, hidden_dim=hidden_dim)
            else:
                raise ValueError(f"Unsupported t_couple_parametrization={self.t_couple_parametrization}")

            self._t_couple_bias = nn.Parameter(torch.zeros(self.steps), requires_grad=self.solver_config.t_couple_requires_grad)
            # placeholder (will be replaced per-batch)
            self.t_couple = torch.zeros(self.steps)

        solver.t_couple = self.t_couple

        # init coef (a/c) parametrizations
        self.a_parametrization = getattr(self.solver_config, "a_parametrization", "diff")
        self.c_parametrization = getattr(self.solver_config, "c_parametrization", "diff")
        self.a_diff_model = None
        self.c_diff_model = None
        self._a_bias = None
        self._c_bias = None

        if self.a_parametrization != "diff":
            out_dim = self.order * self.steps
            if self.a_parametrization == "film_mlp_diff":
                hidden_dim = int(getattr(self.solver_config, "a_film_hidden_dim", 256))
                self.a_diff_model = PromptNoiseFiLMMlp(out_dim=out_dim, hidden_dim=hidden_dim)
            elif self.a_parametrization == "transformer":
                a_cfg = getattr(config, "a_scheduler_model", None)
                if a_cfg is None:
                    a_cfg = config.scheduler_model
                self.a_diff_model = instantiate(a_cfg, num_timesteps=out_dim)
            elif self.a_parametrization == "film_mlp":
                a_cfg = getattr(config, "a_scheduler_model", None)
                if a_cfg is None:
                    a_cfg = config.scheduler_model
                self.a_diff_model = instantiate(a_cfg)
            elif self.a_parametrization == "diff_transformer":
                # ⭐ Новая параметризация: diff + transformer residual
                a_cfg = getattr(config, "a_scheduler_model", None)
                if a_cfg is None:
                    a_cfg = config.scheduler_model
                self.a_diff_model = instantiate(a_cfg, num_timesteps=out_dim)
            else:
                raise ValueError(f"Unsupported a_parametrization={self.a_parametrization}")
            self._a_bias = nn.Parameter(
                torch.zeros(self.order, self.steps),
                requires_grad=self.solver_config.a_requires_grad,
            )

        if self.c_parametrization != "diff":
            out_dim = self.order * self.steps
            if self.c_parametrization == "film_mlp_diff":
                hidden_dim = int(getattr(self.solver_config, "c_film_hidden_dim", 256))
                self.c_diff_model = PromptNoiseFiLMMlp(out_dim=out_dim, hidden_dim=hidden_dim)
            elif self.c_parametrization == "transformer":
                c_cfg = getattr(config, "c_scheduler_model", None)
                if c_cfg is None:
                    c_cfg = config.scheduler_model
                self.c_diff_model = instantiate(c_cfg, num_timesteps=out_dim)
            elif self.c_parametrization == "film_mlp":
                c_cfg = getattr(config, "c_scheduler_model", None)
                if c_cfg is None:
                    c_cfg = config.scheduler_model
                self.c_diff_model = instantiate(c_cfg)
            elif self.c_parametrization == "diff_transformer":
                # ⭐ Новая параметризация: diff + transformer residual
                c_cfg = getattr(config, "c_scheduler_model", None)
                if c_cfg is None:
                    c_cfg = config.scheduler_model
                self.c_diff_model = instantiate(c_cfg, num_timesteps=out_dim)
            else:
                raise ValueError(f"Unsupported c_parametrization={self.c_parametrization}")
            self._c_bias = nn.Parameter(
                torch.zeros(self.order, self.steps),
                requires_grad=self.solver_config.c_requires_grad,
            )

        # baseline coefficients (current behavior)
        # ====== A coefficients ======
        if self.a_parametrization in ("diff", "film_mlp_diff", "diff_transformer"):
            for i in range(1, self.order + 1):
                aname = f'a{i}_diff'
                self.register_parameter(
                    name=aname,
                    param=nn.Parameter(
                        torch.zeros(self.steps),
                        requires_grad=self.solver_config.a_requires_grad
                    )
                )
                solver.__setattr__(aname, self.__getattr__(aname))
        else:
            # placeholder, будет заменён батчевыми coeffs
            for i in range(1, self.order + 1):
                solver.__setattr__(f'a{i}_diff', torch.zeros(self.steps))


        # ====== C coefficients ======
        if self.c_parametrization in ("diff", "film_mlp_diff", "diff_transformer"):
            for i in range(1, self.order + 1):
                cname = f'c{i}_diff'
                self.register_parameter(
                    name=cname,
                    param=nn.Parameter(
                        torch.zeros(self.steps),
                        requires_grad=self.solver_config.c_requires_grad
                    )
                )
                solver.__setattr__(cname, self.__getattr__(cname))
        else:
            for i in range(1, self.order + 1):
                solver.__setattr__(f'c{i}_diff', torch.zeros(self.steps))

        # theory coef
        solver.use_theory_coef = self.solver_config.use_theory_coef
        if not solver.use_theory_coef:
            solver.init_coefs(
                steps=self.steps,
                order=self.order,
                timesteps=self.get_t_steps()
            )
        
        # ==========================================
        # Apply requires_grad flags consistently
        # ==========================================

        def set_requires_grad(module, flag: bool):
            if module is None:
                return
            for p in module.parameters():
                p.requires_grad = flag

        # ---- A parameters ----
        if self.a_parametrization in ("diff", "film_mlp_diff"):
            # diff params already created with requires_grad flag
            pass
        else:
            # control conditional model
            set_requires_grad(self.a_diff_model, self.solver_config.a_requires_grad)
            if self._a_bias is not None:
                self._a_bias.requires_grad = self.solver_config.a_requires_grad


        # ---- C parameters ----
        if self.c_parametrization in ("diff", "film_mlp_diff"):
            pass
        else:
            set_requires_grad(self.c_diff_model, self.solver_config.c_requires_grad)
            if self._c_bias is not None:
                self._c_bias.requires_grad = self.solver_config.c_requires_grad

        if self.stepwise_predictor is not None:
            a_grad = bool(getattr(self.solver_config, "a_requires_grad", True))
            c_grad = bool(getattr(self.solver_config, "c_requires_grad", True))
            set_requires_grad(self.stepwise_predictor, a_grad or c_grad)

        # ---- t_couple parameters ----
        if self.t_couple_parametrization == "diff":
            pass
        else:
            set_requires_grad(self.t_couple_model, self.solver_config.t_couple_requires_grad)
            if self._t_couple_bias is not None:
                self._t_couple_bias.requires_grad = self.solver_config.t_couple_requires_grad
            
        # ==========================================
        # Residual initialization for conditional a/c
        # ==========================================

        def zero_last_layer(module):
            if _final_linear_scheduler_transformer(module) is not None:
                _init_scheduler_transformer_residual_head(module)
                return
            if isinstance(module, nn.Sequential):
                last = module[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    if last.bias is not None:
                        nn.init.zeros_(last.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # A model
        if self.a_diff_model is not None:
            if self.a_parametrization == "diff_transformer":
                # ⭐ Для transformer используем специальную инициализацию
                if hasattr(config, "a_scheduler_model"):
                    t_unif = torch.linspace(1.0, self.t_eps, self.steps + 1).flip(0)
                    # Создаём фейковые логиты для transformer (он ожидает другой формат)
                    # Трансформер будет предсказывать residual, начиная с нуля
                    _init_scheduler_transformer_residual_head(self.a_diff_model)
                else:
                    zero_last_layer(self.a_diff_model)
            else:
                zero_last_layer(self.a_diff_model)

        # C model
        if self.c_diff_model is not None:
            if self.c_parametrization == "diff_transformer":
                # ⭐ Для transformer используем специальную инициализацию
                if hasattr(config, "c_scheduler_model"):
                    _init_scheduler_transformer_residual_head(self.c_diff_model)
                else:
                    zero_last_layer(self.c_diff_model)
            else:
                zero_last_layer(self.c_diff_model)

        # Расписание t: inv stick-breaking для равномерной сетки
        if self.t_parametrization == "transformer":
            t_unif = torch.linspace(1.0, self.t_eps, self.steps + 1).flip(0)
            inv_logits = self.get_inv_t_steps(t_unif)
            _init_scheduler_transformer_residual_head(self.mu_logit, logistic_bias_1d=inv_logits)
        
        # end init solver
        self.solver = solver
        
    def disable_mlp(self):
        """Отключает MLP-добавку (используются только базовые diff параметры)."""
        self.mlp_disabled = True

    def enable_mlp(self):
        """Включает MLP-добавку."""
        self.mlp_disabled = False

    # Для обратной совместимости с training.py (если там вызывается freeze_mlp/unfreeze_mlp)
    def freeze_mlp(self):
        self.disable_mlp()

    def unfreeze_mlp(self):
        self.enable_mlp()
        
    def _get_shuffled_inputs_for_coefficients(
        self, 
        x_t: torch.Tensor, 
        cond_emb: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Create shuffled inputs ONLY for coefficient prediction.
        Does NOT affect the denoiser.
        """
        if not (self.shuffle_coef_noise or self.shuffle_coef_conditioning):
            return x_t, cond_emb

        B = x_t.shape[0]
        shuffled_x_t = x_t
        shuffled_cond_emb = cond_emb

        if self.shuffle_coef_noise:
            noise_perm = torch.randperm(B, device=x_t.device)
            shuffled_x_t = x_t[noise_perm]

        if self.shuffle_coef_conditioning and cond_emb is not None:
            cond_perm = torch.randperm(B, device=cond_emb.device)
            shuffled_cond_emb = cond_emb[cond_perm]

        return shuffled_x_t, shuffled_cond_emb

    def _clear_stepwise_sampling_state(self) -> None:
        self._stepwise_coeff_buffers = None
        self._stepwise_sampling_timesteps = None

    def _a_has_diff_baseline(self) -> bool:
        return self.a_parametrization in ("diff", "film_mlp_diff", "diff_transformer")

    def _c_has_diff_baseline(self) -> bool:
        return self.c_parametrization in ("diff", "film_mlp_diff", "diff_transformer")

    def _init_coeff_buffers(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """Allocate (B, steps) coefficient tables for stepwise sampling."""
        buffers: Dict[str, torch.Tensor] = {}
        for i in range(1, self.order + 1):
            a_name = f"a{i}_diff"
            if self._a_has_diff_baseline():
                base_a = getattr(self, a_name)
                buffers[a_name] = base_a.unsqueeze(0).expand(batch_size, -1).clone().to(device=device, dtype=dtype)
            else:
                buffers[a_name] = torch.zeros(batch_size, self.steps, device=device, dtype=dtype)

            c_name = f"c{i}_diff"
            if self._c_has_diff_baseline():
                base_c = getattr(self, c_name)
                buffers[c_name] = base_c.unsqueeze(0).expand(batch_size, -1).clone().to(device=device, dtype=dtype)
            else:
                buffers[c_name] = torch.zeros(batch_size, self.steps, device=device, dtype=dtype)

        self._stepwise_coeff_buffers = buffers
        self._assign_stepwise_buffers_to_solver()

    def _assign_stepwise_buffers_to_solver(self) -> None:
        if self._stepwise_coeff_buffers is None:
            return
        for name, buf in self._stepwise_coeff_buffers.items():
            self.solver.__setattr__(name, buf)

    def _get_step_time(
        self,
        step_idx: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        timesteps = self._stepwise_sampling_timesteps
        if timesteps is None:
            return torch.zeros(batch_size, device=device, dtype=dtype)
        if timesteps.ndim == 2:
            return timesteps[:, step_idx].to(device=device, dtype=dtype)
        return timesteps[step_idx].expand(batch_size).to(device=device, dtype=dtype)

    def _update_stepwise_coeffs(
        self,
        step_idx: int,
        x_t: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> None:
        """Predict and attach coefficients for the current solver step (no in-place buffer writes)."""
        batch_size = x_t.shape[0]
        device, dtype = x_t.device, x_t.dtype

        if self.mlp_disabled or self.stepwise_predictor is None:
            if self._stepwise_coeff_buffers is None:
                self._init_coeff_buffers(batch_size, device, dtype)
            return

        shuffled_x, shuffled_cond = self._get_shuffled_inputs_for_coefficients(x_t, cond_emb)
        if shuffled_cond is None:
            return

        t_current = self._get_step_time(step_idx, batch_size, device, dtype)
        a_pred, c_pred = self.stepwise_predictor(shuffled_x, t_current, shuffled_cond)

        # Rebuild (B, steps) tables each call. _get_step_param reads column params_step;
        # broadcasting the current-step value avoids in-place mutation that breaks backward.
        buffers: Dict[str, torch.Tensor] = {}
        for i in range(1, self.order + 1):
            a_name = f"a{i}_diff"
            c_name = f"c{i}_diff"
            if self._a_has_diff_baseline():
                a_val = getattr(self, a_name)[step_idx] + a_pred[:, i - 1]
            else:
                a_val = a_pred[:, i - 1]
            if self._c_has_diff_baseline():
                c_val = getattr(self, c_name)[step_idx] + c_pred[:, i - 1]
            else:
                c_val = c_pred[:, i - 1]
            buffers[a_name] = a_val.unsqueeze(1).expand(batch_size, self.steps)
            buffers[c_name] = c_val.unsqueeze(1).expand(batch_size, self.steps)

        self._stepwise_coeff_buffers = buffers
        self._assign_stepwise_buffers_to_solver()

    def _update_dynamic_t_couple(
        self,
        noise: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> None:
        if self.t_couple_parametrization == "diff":
            return

        shuffled_noise, shuffled_cond_emb = self._get_shuffled_inputs_for_coefficients(noise, cond_emb)
        if shuffled_cond_emb is not None:
            t_couple = self.t_couple_model(shuffled_noise, shuffled_cond_emb)
        else:
            t_couple = self._t_couple_bias.reshape(1, self.steps)
        self.solver.t_couple = t_couple

    def _update_dynamic_ac_coefs(
        self,
        noise: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> None:
        """
        If enabled by config, compute batch-dependent a/c coefficient tables and
        attach them to the solver as tensors of shape (B, steps).
        """
        # ---- SHARED BACKBONE ----
        if self.use_shared_ac_backbone:
            if self.mlp_disabled:
                # Только базовые параметры (diff)
                if self.a_parametrization in ("film_mlp_diff", "diff_transformer"):
                    base_a = torch.stack([getattr(self, f'a{i}_diff') for i in range(1, self.order+1)], dim=0)
                    a_all = base_a.unsqueeze(0).expand(noise.shape[0], -1, -1)
                else:
                    a_all = self._a_bias.unsqueeze(0).expand(noise.shape[0], -1, -1)
                for i in range(1, self.order+1):
                    self.solver.__setattr__(f"a{i}_diff", a_all[:, i-1, :])

                if self.c_parametrization in ("film_mlp_diff", "diff_transformer"):
                    base_c = torch.stack([getattr(self, f'c{i}_diff') for i in range(1, self.order+1)], dim=0)
                    c_all = base_c.unsqueeze(0).expand(noise.shape[0], -1, -1)
                else:
                    c_all = self._c_bias.unsqueeze(0).expand(noise.shape[0], -1, -1)
                for i in range(1, self.order+1):
                    self.solver.__setattr__(f"c{i}_diff", c_all[:, i-1, :])
                return

            # Активный режим
            if cond_emb is not None:
                shuffled_noise, shuffled_cond_emb = self._get_shuffled_inputs_for_coefficients(noise, cond_emb)
                a_flat, c_flat = self.shared_ac_backbone(shuffled_noise, shuffled_cond_emb)
            else:
                a_flat = self._a_bias.reshape(1, -1).expand(noise.shape[0], -1)
                c_flat = self._c_bias.reshape(1, -1).expand(noise.shape[0], -1)

            # Применяем a
            if self.a_parametrization in ("film_mlp_diff", "diff_transformer"):
                base_a = torch.stack([getattr(self, f'a{i}_diff') for i in range(1, self.order+1)], dim=0)
                a_all = base_a.unsqueeze(0) + a_flat.reshape(-1, self.order, self.steps)
            else:
                a_all = a_flat.reshape(noise.shape[0], self.order, self.steps)
            for i in range(1, self.order+1):
                self.solver.__setattr__(f"a{i}_diff", a_all[:, i-1, :])

            # Применяем c
            if self.c_parametrization in ("film_mlp_diff", "diff_transformer"):
                base_c = torch.stack([getattr(self, f'c{i}_diff') for i in range(1, self.order+1)], dim=0)
                c_all = base_c.unsqueeze(0) + c_flat.reshape(-1, self.order, self.steps)
            else:
                c_all = c_flat.reshape(noise.shape[0], self.order, self.steps)
            for i in range(1, self.order+1):
                self.solver.__setattr__(f"c{i}_diff", c_all[:, i-1, :])
            return
        
        if self.a_parametrization not in ("diff", "film_mlp_diff", "diff_transformer"):
            out_dim = self.order * self.steps
            if self.a_parametrization in ("film_mlp", "prompt_noise_film_mlp"):
                shuffled_noise, shuffled_cond_emb = self._get_shuffled_inputs_for_coefficients(noise, cond_emb)
                if shuffled_cond_emb is not None:
                    a_flat = self.a_diff_model(shuffled_noise, shuffled_cond_emb)
                else:
                    a_flat = self._a_bias.reshape(1, out_dim)
            else:  # transformer
                if cond_emb is not None:
                    a_flat = self.a_diff_model(noise, cond_emb)  # (B, out_dim)
                else:
                    dev = noise.device
                    dummy_cond = torch.zeros(1, 77, 768, device=dev)
                    a_flat = self.a_diff_model(noise, dummy_cond)  # (1, out_dim)
            a_all = a_flat.reshape(a_flat.shape[0], self.order, self.steps)
            
            for i in range(1, self.order + 1):
                self.solver.__setattr__(f"a{i}_diff", a_all[:, i - 1, :])
        elif self.a_parametrization == "film_mlp_diff":
            # ⭐ Режим diff + residual из MLP
            base_a = torch.stack([getattr(self, f'a{i}_diff') for i in range(1, self.order+1)], dim=0)

            # ⭐ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: проверяем mlp_disabled вместо mlp_frozen
            if not self.mlp_disabled:
                # MLP включен: base + mlp_residual
                if cond_emb is not None:
                    mlp_out = self.a_diff_model(noise, cond_emb)
                else:
                    mlp_out = self._a_bias.reshape(1, -1).expand(noise.shape[0], -1)
                mlp_out = mlp_out.reshape(-1, self.order, self.steps)
                a_all = base_a.unsqueeze(0) + mlp_out
            else:
                # ⭐ MLP отключен: используем только base_a (diff параметры)
                # Градиенты всё равно текут через весь граф!
                a_all = base_a.unsqueeze(0).expand(noise.shape[0], -1, -1)

            for i in range(1, self.order+1):
                self.solver.__setattr__(f"a{i}_diff", a_all[:, i-1, :])
        elif self.a_parametrization == "diff_transformer":
            # ⭐ Режим diff + transformer residual
            base_a = torch.stack([getattr(self, f'a{i}_diff') for i in range(1, self.order+1)], dim=0)

            if not self.mlp_disabled:
                # Transformer включен: base + transformer_residual
                if cond_emb is not None:
                    transformer_out = self.a_diff_model(noise, cond_emb)
                else:
                    dev = noise.device
                    dummy_cond = torch.zeros(1, 77, 768, device=dev, dtype=noise.dtype)
                    transformer_out = self.a_diff_model(noise, dummy_cond)
                transformer_out = transformer_out.reshape(-1, self.order, self.steps)
                a_all = base_a.unsqueeze(0) + transformer_out
            else:
                # ⭐ Transformer отключен: используем только base_a (diff параметры)
                a_all = base_a.unsqueeze(0).expand(noise.shape[0], -1, -1)

            for i in range(1, self.order+1):
                self.solver.__setattr__(f"a{i}_diff", a_all[:, i-1, :])
        
        if self.c_parametrization not in ("diff", "film_mlp_diff", "diff_transformer"):
            out_dim = self.order * self.steps
            if self.c_parametrization in ("film_mlp", "prompt_noise_film_mlp"):
                shuffled_noise, shuffled_cond_emb = self._get_shuffled_inputs_for_coefficients(noise, cond_emb)
                if shuffled_cond_emb is not None:
                    c_flat = self.c_diff_model(shuffled_noise, shuffled_cond_emb)
                else:
                    c_flat = self._c_bias.reshape(1, out_dim)
            else:  # transformer
                if cond_emb is not None:
                    c_flat = self.c_diff_model(noise, cond_emb)  # (B, out_dim)
                else:
                    dev = noise.device
                    dummy_cond = torch.zeros(1, 77, 768, device=dev)
                    c_flat = self.c_diff_model(noise, dummy_cond)  # (1, out_dim)
            c_all = c_flat.reshape(c_flat.shape[0], self.order, self.steps)
            
            for i in range(1, self.order + 1):
                self.solver.__setattr__(f"c{i}_diff", c_all[:, i - 1, :])
        elif self.c_parametrization == "film_mlp_diff":  # ⭐ ИСПРАВЛЕНО: было a_parametrization
            # ⭐ Режим diff + residual из MLP
            base_c = torch.stack([getattr(self, f'c{i}_diff') for i in range(1, self.order+1)], dim=0)

            # ⭐ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: проверяем mlp_disabled вместо mlp_frozen
            if not self.mlp_disabled:
                # MLP включен: base + mlp_residual
                if cond_emb is not None:
                    mlp_out = self.c_diff_model(noise, cond_emb)
                else:
                    mlp_out = self._c_bias.reshape(1, -1).expand(noise.shape[0], -1)
                mlp_out = mlp_out.reshape(-1, self.order, self.steps)
                c_all = base_c.unsqueeze(0) + mlp_out
            else:
                # ⭐ MLP отключен: используем только base_c (diff параметры)
                # Градиенты всё равно текут через весь граф!
                c_all = base_c.unsqueeze(0).expand(noise.shape[0], -1, -1)

            for i in range(1, self.order+1):
                self.solver.__setattr__(f"c{i}_diff", c_all[:, i-1, :])
        elif self.c_parametrization == "diff_transformer":
            # ⭐ Режим diff + transformer residual
            base_c = torch.stack([getattr(self, f'c{i}_diff') for i in range(1, self.order+1)], dim=0)

            if not self.mlp_disabled:
                # Transformer включен: base + transformer_residual
                if cond_emb is not None:
                    transformer_out = self.c_diff_model(noise, cond_emb)
                else:
                    dev = noise.device
                    dummy_cond = torch.zeros(1, 77, 768, device=dev, dtype=noise.dtype)
                    transformer_out = self.c_diff_model(noise, dummy_cond)
                transformer_out = transformer_out.reshape(-1, self.order, self.steps)
                c_all = base_c.unsqueeze(0) + transformer_out
            else:
                # ⭐ Transformer отключен: используем только base_c (diff параметры)
                c_all = base_c.unsqueeze(0).expand(noise.shape[0], -1, -1)

            for i in range(1, self.order+1):
                self.solver.__setattr__(f"c{i}_diff", c_all[:, i-1, :])

    # timesteps logic
    def get_t_steps(self, noise=None, cond_emb=None, **kwargs) -> torch.Tensor:
        """Get generation timesteps."""
        if self.t_parametrization in ("diff", "mu_logit"):
            logits = self.mu_logit
        elif self.t_parametrization in ("film_mlp", "prompt_noise_film_mlp"):
            if cond_emb is not None:
                logits = self.mu_logit(noise, cond_emb)  # (B, steps-1)
            else:
                logits = torch.zeros(1, self.steps - 1, device=noise.device, dtype=noise.dtype)
        elif self.t_parametrization == "transformer":
            if cond_emb is not None:
                logits = self.mu_logit(noise, cond_emb)
            else:
                logits = self.mu_logit(noise, torch.zeros(1, 77, 768, device=noise.device, dtype=noise.dtype))[0]
                
        t = self.get_mu_t_steps(logits)
        # keep the same direction as before
        if t.ndim == 2:
            return t.flip(1)
        return t.flip(0)
    
    def get_mu_t_steps(self, mu_logit: torch.Tensor) -> torch.Tensor:
        """Use stick-breaking transform for getting timesteps from logits.
        Timesteps are calculated following Eq. 14 from the GAS paper.
        """
        t_offset = self.t_eps

        mu = mu_logit.sigmoid()
        mu = mu * (1 - 2 * self.eps_mu_offset) + self.eps_mu_offset

        if mu.ndim == 2:
            # (B, steps-1) -> cumprod over step dimension
            t_steps = 1 - torch.cumprod(mu, dim=1)
            t_steps = t_steps * (1 - t_offset) + t_offset
            t_steps = torch.cat(
                [
                    torch.zeros_like(t_steps[:, :1]) + t_offset,
                    t_steps,
                    torch.ones_like(t_steps[:, :1]),
                ],
                dim=1,
            )
            return t_steps

        t_steps = 1 - torch.cumprod(mu, 0)
        t_steps = t_steps * (1 - t_offset) + t_offset
        t_steps = torch.cat(
            [
                torch.zeros_like(t_steps[:1]) + t_offset,
                t_steps,
                torch.ones_like(t_steps[:1])
            ]
        )
        return t_steps
    
    def get_inv_t_steps(self, t_steps) -> torch.Tensor:
        """Function to inverse initialized timesteps."""
        t_steps = t_steps[1:-1]
        t_steps = 1 - (t_steps - self.t_eps) / (1 - self.t_eps)
        t_steps = t_steps / torch.concat([torch.ones_like(t_steps[:1]), t_steps[:-1]])
        t_steps = (t_steps - self.eps_mu_offset) / (1 - 2 * self.eps_mu_offset)

        return t_steps.logit()
    
    # utilities
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Loads EMA parameters checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        ema = ExponentialMovingAverage(self.parameters(), 0.1)
        ema.load_state_dict(checkpoint['ema'])
        ema.copy_to(self.parameters())

    # def parameters(self) -> List[nn.parameter.Parameter]:
    #     """Returns list of specified solver and wrapper parameters."""
    #     return list(p for p in super().parameters() if p.requires_grad)

    def interpolate_lpips(self, x: torch.Tensor) -> torch.Tensor:
        """Utility function to resize images for LPIPS calculation."""
        return interpolate(x, size=224, mode='bilinear').clip(-1., 1.)

    # solvers
    def get_base_solver(self) -> GeneralizedSolver:
        """Initialises Generalized Solver from model_fn 
        and noise scheduler of the BaseModel instance.
        """
        solver = GeneralizedSolver(
            model_fn=self.model.model_fn,
            noise_schedule=self.model.ns,
        )
        return solver

    @staticmethod
    def _align_pred_to_gt(pred: torch.Tensor, gt: torch.Tensor) -> Optional[torch.Tensor]:
        """Broadcast predicted solver tensor to ground-truth shape when possible."""
        pred = pred.to(device=gt.device, dtype=gt.dtype)
        if pred.shape == gt.shape:
            return pred
        try:
            return torch.broadcast_to(pred, gt.shape)
        except RuntimeError:
            return None

    def _predicted_gs_solver_tensors(
        self,
        noise: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Snapshot coefficient tensors currently attached to the solver (same storage as teacher pickle)."""
        out: Dict[str, torch.Tensor] = {}
        for i in range(1, self.order + 1):
            for prefix in ("a", "c"):
                name = f"{prefix}{i}_diff"
                t = getattr(self.solver, name, None)
                if isinstance(t, torch.Tensor):
                    out[name] = t
        tc = getattr(self.solver, "t_couple", None)
        if isinstance(tc, torch.Tensor):
            out["t_couple"] = tc
        ts = self.solver.get_time_steps(noise, cond_emb)
        out["timesteps"] = ts
        return out

    def _add_gt_solver_metrics(
        self,
        d: dict,
        gt_solver_params: Optional[Dict[str, torch.Tensor]],
        noise: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> None:
        """Per-batch-element MSE/MAE vs teacher ``manual_solver_params`` when present."""
        if not gt_solver_params:
            return
        pred_all = self._predicted_gs_solver_tensors(noise, cond_emb)
        mse_list: List[torch.Tensor] = []
        mae_list: List[torch.Tensor] = []
        for name, gt in gt_solver_params.items():
            if name not in pred_all:
                continue
            aligned = self._align_pred_to_gt(pred_all[name], gt)
            if aligned is None:
                continue
            diff = aligned - gt
            mse = diff.flatten(start_dim=1).pow(2).mean(dim=1)
            mae = diff.flatten(start_dim=1).abs().mean(dim=1)
            d[f"gt_mse_{name}"] = mse
            d[f"gt_mae_{name}"] = mae
            mse_list.append(mse)
            mae_list.append(mae)
        if mse_list:
            d["gt_mse_mean"] = torch.stack(mse_list, dim=0).mean(dim=0)
            d["gt_mae_mean"] = torch.stack(mae_list, dim=0).mean(dim=0)
    
    @contextlib.contextmanager
    def _manual_solver_params_context(self, manual_solver_params: Optional[dict]):
        if not manual_solver_params:
            yield
            return

        original_values = {}
        for key, value in manual_solver_params.items():
            if key == "timesteps":
                continue
            if hasattr(self.solver, key):
                original_values[key] = getattr(self.solver, key)
            setattr(self.solver, key, value)

        original_get_time_steps = self.solver.get_time_steps
        if "timesteps" in manual_solver_params and manual_solver_params["timesteps"] is not None:
            manual_timesteps = manual_solver_params["timesteps"]
            self.solver.get_time_steps = lambda *args, **kwargs: manual_timesteps

        try:
            yield
        finally:
            self.solver.get_time_steps = original_get_time_steps
            for key, value in original_values.items():
                setattr(self.solver, key, value)

    def student_sampler_fn(self, noise: torch.Tensor, **kwargs) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Calls `sample` method of the Generalised Solver. 
        
        Args:
            noise (torch.Tensor): An initial noise tensor to start sampling process from.
        
        Returns:
            None: A placeholder for consistency with latent models.
            torch.tensor: Sampled images
        """
        cond_emb = getattr(self.model.model_fn, "condition", None)
        manual_solver_params = kwargs.pop("manual_solver_params", None)
            
        with self._manual_solver_params_context(manual_solver_params):
            before_step_fn = None
            if manual_solver_params is None:
                if self.use_stepwise_coeff:
                    self._stepwise_sampling_timesteps = self.solver.get_time_steps(noise, cond_emb)
                    self._stepwise_coeff_buffers = None
                    self._update_dynamic_t_couple(noise=noise, cond_emb=cond_emb)

                    def before_step_fn(step_idx: int, x: torch.Tensor) -> None:
                        self._update_stepwise_coeffs(step_idx, x, cond_emb)
                else:
                    self._update_dynamic_t_couple(noise=noise, cond_emb=cond_emb)
                    self._update_dynamic_ac_coefs(noise=noise, cond_emb=cond_emb)

            try:
                images = self.solver.sample(
                    x=noise,
                    steps=self.steps,
                    order=self.order,
                    before_step_fn=before_step_fn,
                )
            finally:
                self._clear_stepwise_sampling_state()
        return None, images
    
    # training function
    def forward(self, batch: SyntDataType, return_timesteps: bool = False, is_train: bool = True) -> dict:
        """Forward function used in training loop. Evaluates solver and calculates losses.
        
        Args:
            batch (SyntDataType): Dataset tuple of size 4. 
                First two arguments are treated like torch.Tensor noise and images samples.
                Second two arguments are optional and can be used in GSWrapperLatent for latent diffusion models.
                They are treated as latents tensors and conditions.
            return_timesteps (bool): Flag whether to return timestep of the current step.
            is_train (bool): Flag whether forward is called in the train loop. 
                Used in `discriminator_step` method of an DistAdversarialTraining instance.
            
        Returns:
            dict: Dictionary of all losses and model outputs. 
                Has `loss_total` key as a weighted sum of adversarial and distillation losses.
        """
        assert len(batch) in (4, 5), f"len(batch) expected 4 or 5, got {len(batch)}"
        gt_solver_params = batch[4] if len(batch) == 5 else None
        noise, images, _, _ = batch[:4]

        d = {}
        if return_timesteps:
            # For conditional t parametrizations (e.g. film_mlp), timesteps depend on cond_emb (+ noise),
            # so pass them explicitly (same behavior as in GSWrapperLatent).
            cond_emb = getattr(self.model.model_fn, "condition", None)
            d['timesteps'] = self.solver.get_time_steps(noise, cond_emb)
        _, student_images = self.student_sampler_fn(noise)
        cond_emb = getattr(self.model.model_fn, "condition", None)
        self._add_gt_solver_metrics(d, gt_solver_params, noise, cond_emb)

        d['loss_l1'] = torch.abs(student_images - images).mean((1, 2, 3))
        d['loss_l2'] = torch.square(student_images - images).mean((1, 2, 3))

        d['x0_s'] = self.interpolate_lpips(student_images)
        d['x0_t'] = self.interpolate_lpips(images)

        d['loss_lpips'] = self.loss_fn_vgg(d['x0_s'], d['x0_t']).flatten(0)

        if self.loss_config.loss_type == 'GAS':
            # disctiminator step optim
            with torch.no_grad():
                _, student_images_disc = self.student_sampler_fn(
                    torch.randn_like(noise)
                )
            res = self.adv_loss.discriminator_step(
                FakeSamples=student_images_disc,
                RealSamples=images,
                is_train=is_train
            )
            d['dis_loss_adv'] = res[0]
            d['dis_scores_fake'] = res[1]
            d['dis_signs_fake'] = res[1].sign()
            d['dis_r1'] = res[2]
            d['dis_r2'] = res[3]

            # generator step optim
            loss_adv, res = self.adv_loss.AccumulateGeneratorGradients(
                FakeSamples=student_images,
                RealSamples=images
            )
            d['gen_loss_adv'] = loss_adv
            d['gen_fake_gen'] = res[1]
            d['gen_signs_fake'] = res[1].sign()

            assert d['gen_loss_adv'].shape == d[self.loss_config.loss_key].shape, f"""
                Shape of generator loss is not equal to distillation loss shape. 
                ({d['gen_loss_adv'].shape} vs {d[self.loss_config.loss_key].shape}).
            """

        d['loss_total'] = self.loss_config.disc_weight * d.get('gen_loss_adv', 0.) + d[self.loss_config.loss_key]

        return d
    
    
class GSWrapperLatent(GSWrapper):
    """Generalised Solver wrapper adapted for latent models."""
    def __init__(self, model: nn.Module, config: ConfigDict):
        super().__init__(model=model, config=config)

    def student_sampler_fn(
        self,
        noise: torch.Tensor,
        decode: bool = False,
        condition: Any = None,
        manual_solver_params: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Calls `sample` method of the Generalised Solver. 
        
        Args:
            noise (torch.Tensor): An initial noise tensor to start sampling process from.
        
        Returns:
            torch.Tensor: Predicted latents that are the direct output of the model.
            Optional[torch.Tensor]: Predicted images (decoded latents).
                Not None if decode flag is set True.
        """
        images = None
        if condition is not None:
            self.model.set_condition(condition)
        cond_emb = getattr(self.model.model_fn, "condition", None)
        
        with self._manual_solver_params_context(manual_solver_params):
            before_step_fn = None
            if manual_solver_params is None:
                if self.use_stepwise_coeff:
                    self._stepwise_sampling_timesteps = self.solver.get_time_steps(noise, cond_emb)
                    self._stepwise_coeff_buffers = None
                    self._update_dynamic_t_couple(noise=noise, cond_emb=cond_emb)

                    def before_step_fn(step_idx: int, x: torch.Tensor) -> None:
                        self._update_stepwise_coeffs(step_idx, x, cond_emb)
                else:
                    self._update_dynamic_t_couple(noise=noise, cond_emb=cond_emb)
                    self._update_dynamic_ac_coefs(noise=noise, cond_emb=cond_emb)

            try:
                latents = self.solver.sample(
                    x=noise,
                    steps=self.steps,
                    order=self.order,
                    before_step_fn=before_step_fn,
                )
            finally:
                self._clear_stepwise_sampling_state()

        if decode:
            images = self.model.decode(latents)

        return latents, images
    
    def forward(self, batch: SyntDataType, return_timesteps: bool = False, is_train: bool = True) -> dict:
        assert len(batch) in (4, 5), f"len(batch) expected 4 or 5, got {len(batch)}"
        gt_solver_params = batch[4] if len(batch) == 5 else None
        noise, images, latents, condition = batch[:4]

        d = {}
        if return_timesteps:
            if condition is not None:
                self.model.set_condition(condition)
            cond_emb = self.model.model_fn.condition
            d['timesteps'] = self.solver.get_time_steps(noise, cond_emb)
        student_latents, _ = self.student_sampler_fn(
            noise,
            condition=condition
        )
        cond_emb = getattr(self.model.model_fn, "condition", None)
        self._add_gt_solver_metrics(d, gt_solver_params, noise, cond_emb)

        d['loss_l1_latents'] = torch.abs(latents - student_latents).mean((1, 2, 3))
        d['loss_l2_latents'] = torch.square(latents - student_latents).mean((1, 2, 3))
        d['x0_t'] = self.interpolate_lpips(images)
        d['latents_s'] = student_latents

        apply_repa = self.use_repa and self.repa_loss is not None and (
            is_train or self.use_repa_in_eval
        )
        if apply_repa:
            dec_grad = getattr(self.loss_config, "repa_grad_through_decoder", True) if is_train else False
            dec_ctx = torch.enable_grad() if dec_grad else torch.no_grad()
            with dec_ctx:
                student_images = self.model.decode(student_latents)
            d["x0_s"] = self.interpolate_lpips(student_images)
            log_repa_layers = getattr(self.loss_config, "repa_log_per_layer", False)
            loss_repa, repa_info = self.repa_loss(
                student_images=student_images,
                teacher_images=images,
                return_detailed=log_repa_layers,
            )
            d["loss_repa"] = loss_repa
            if repa_info is not None:
                for rk, rv in repa_info.items():
                    d[rk] = rv

        if self.loss_config.loss_type == "GAS":
            with torch.no_grad():
                student_latents_disc, _ = self.student_sampler_fn(
                    torch.randn_like(noise)
                )
            res = self.adv_loss.discriminator_step(
                FakeSamples=student_latents_disc,
                RealSamples=latents,
                is_train=is_train
            )

            d['dis_loss_adv'] = res[0]
            d['dis_scores_fake'] = res[1]
            d['dis_signs_fake'] = res[1].sign()
            d['dis_r1'] = res[2]
            d['dis_r2'] = res[3]

            # generator step
            loss_adv, res = self.adv_loss.AccumulateGeneratorGradients(
                FakeSamples=student_latents,
                RealSamples=latents
            )
            d['gen_loss_adv'] = loss_adv
            d['gen_fake_gen'] = res[1]
            d['gen_signs_fake'] = res[1].sign()

            assert d['gen_loss_adv'].shape == d[self.loss_config.loss_key].shape, f"SHAPE = {d['gen_loss_adv'].shape}, {d[self.loss_config.loss_key].shape}"

        base_loss = d[self.loss_config.loss_key]
        adv_part = self.loss_config.disc_weight * d.get("gen_loss_adv", 0.0)
        repa_part = torch.tensor(
            0.0, device=base_loss.device, dtype=base_loss.dtype
        )
        if self.use_repa and self.repa_loss is not None and "loss_repa" in d:
            rw = float(getattr(self.loss_config, "repa_weight", 0.5))
            repa_part = rw * d["loss_repa"].to(device=base_loss.device, dtype=base_loss.dtype)
        d["loss_total"] = adv_part + base_loss + repa_part

        return d
