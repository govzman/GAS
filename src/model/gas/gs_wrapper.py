import lpips

import torch
from torch import nn
from typing import Tuple, Optional, Any, List
from torch.nn.functional import interpolate
from torch_ema import ExponentialMovingAverage
from ml_collections import ConfigDict
from hydra.utils import instantiate

from src.model.gas.base_model import BaseModel
from src.model.gas.generalized_solver import GeneralizedSolver
from src.model.gas.adversarial_module.dist_adv_loss import DistAdversarialTraining
from src.model.gas.synt_data import SyntDataType
from src.model.gas.stepwise_predictors import create_stepwise_predictor


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
    bias — нули или заданный вектор логитов (как для linear/mlp t-расписания), если совпадает размерность.
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


class PromptNoiseFiLMMlp(nn.Module):
    """
    Stable conditional MLP for coefficient tables that depends on both:
    - prompt embedding (cond_emb): (B, T, 768)
    - initial noise (noise): (B, C, H, W) or (B, ...)
    Produces (B, out_dim) where out_dim = order * steps.
    """

    def __init__(self, out_dim: int, prompt_dim: int = 768, hidden_dim: int = 256, noise_feat_dim: int = 3):
        super().__init__()
        self.prompt_norm = nn.LayerNorm(prompt_dim)
        self.noise_norm = nn.LayerNorm(noise_feat_dim)

        self.prompt_mlp = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # FiLM params from noise branch: gamma,beta for prompt features
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # start from near-zero residuals (wrapper further scales residuals)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _noise_stats(noise: torch.Tensor) -> torch.Tensor:
        # Robust low-dim summary of z0. Shape: (B, 3) = [mean, std, rms]
        b = noise.shape[0]
        flat = noise.reshape(b, -1)
        mean = flat.mean(dim=1)
        std = flat.std(dim=1, unbiased=False)
        rms = (flat.pow(2).mean(dim=1) + 1e-12).sqrt()
        return torch.stack([mean, std, rms], dim=1)

    def forward(self, noise: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # cond_emb: (B, T, 768) -> pooled (B, 768)
        p = cond_emb.mean(dim=1)
        # p = self.prompt_norm(p)
        hp = self.prompt_mlp(p)

        n = self._noise_stats(noise).to(dtype=hp.dtype, device=hp.device)
        # n = self.noise_norm(n)
        hn = self.noise_mlp(n)

        gamma, beta = self.film(hn).chunk(2, dim=1)
        # h = self.out_norm(hp * (1.0 + gamma) + beta)
        h = hp * (1.0 + gamma) + beta
        return self.head(h)
    

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
        
        # create lpips
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').requires_grad_(False)
        self.loss_fn_vgg.eval()

        # construct loss
        self.loss_config = config.loss_config
        assert self.loss_config.loss_type in ["GS", "GAS"]
        if self.loss_config.loss_type == "GAS":
            self.adv_loss = DistAdversarialTraining(self.loss_config)

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
        elif self.t_parametrization == "linear":
            self.mu_logit = nn.Linear(in_features=768, out_features=self.steps - 1)
            nn.init.normal_(self.mu_logit.weight, std=0.02)
            # nn.init.kaiming_uniform_(self.mu_logit.weight, a=0.1)
            t_unif = torch.linspace(1., self.t_eps, self.steps + 1).flip(0)
            with torch.no_grad():
                self.mu_logit.bias.copy_(self.get_inv_t_steps(t_unif))
        elif self.t_parametrization == "mlp":
            self.mu_logit = nn.Sequential(
                nn.Linear(in_features=768, out_features=768 * 4, bias=False),
                nn.LeakyReLU(),
                nn.Linear(in_features=768 * 4, out_features=self.steps - 1)
            )
            nn.init.kaiming_uniform_(self.mu_logit[0].weight, a=0.2)
            nn.init.zeros_(self.mu_logit[2].weight)
            t_unif = torch.linspace(1., self.t_eps, self.steps + 1).flip(0)
            with torch.no_grad():
                self.mu_logit[2].bias.copy_(self.get_inv_t_steps(t_unif))
        elif self.t_parametrization in ("film_mlp", "prompt_noise_film_mlp"):
            hidden_dim = int(getattr(self.solver_config, "t_film_hidden_dim", 256))
            self.mu_logit = PromptNoiseFiLMMlp(out_dim=self.steps - 1, hidden_dim=hidden_dim)
        elif self.t_parametrization == "transformer":
            self.mu_logit = instantiate(config.scheduler_model)
        else:
            raise ValueError(f"Unsupported t_parametrization={self.t_parametrization}")

        solver.get_time_steps = lambda *args, **kwargs: self.get_t_steps(*args, **kwargs)

        # init t_couple
        self.t_couple_parametrization = getattr(self.solver_config, "t_couple_parametrization", "diff")
        self.t_couple_model = None
        self._t_couple_bias = None

        if self.t_couple_parametrization == "diff":
            self.t_couple = nn.Parameter(torch.zeros(self.steps), requires_grad=self.solver_config.t_couple_requires_grad)
        else:
            if self.t_couple_parametrization == "linear":
                self.t_couple_model = nn.Linear(in_features=768, out_features=self.steps)
                nn.init.normal_(self.t_couple_model.weight, std=0.02)
                nn.init.zeros_(self.t_couple_model.bias)
            elif self.t_couple_parametrization == "mlp":
                self.t_couple_model = nn.Sequential(
                    nn.Linear(in_features=768, out_features=768 * 4, bias=False),
                    nn.LayerNorm(768 * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=768 * 4, out_features=768 * 4, bias=False),
                    nn.LayerNorm(768 * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=768 * 4, out_features=self.steps),
                )
                nn.init.kaiming_uniform_(self.t_couple_model[0].weight, a=0.2)
                nn.init.kaiming_uniform_(self.t_couple_model[3].weight, a=0.2)
                nn.init.zeros_(self.t_couple_model[6].weight)
                nn.init.zeros_(self.t_couple_model[6].bias)
            elif self.t_couple_parametrization in ("film_mlp", "prompt_noise_film_mlp"):
                hidden_dim = int(getattr(self.solver_config, "t_couple_film_hidden_dim", 256))
                self.t_couple_model = PromptNoiseFiLMMlp(out_dim=self.steps, hidden_dim=hidden_dim)
            elif self.t_couple_parametrization == "stepwise":
                self.t_couple_model = create_stepwise_predictor(
                    out_dim=1,
                    version=getattr(self.solver_config, "t_couple_stepwise_version", "lightweight"),
                    latent_channels=self.latent_channels,
                    text_embed_dim=self.text_embed_dim,
                    encoder_width=int(getattr(self.solver_config, "t_couple_stepwise_encoder_width", 32)),
                    num_conv_blocks=int(getattr(self.solver_config, "t_couple_stepwise_conv_blocks", 2)),
                    attention_heads=int(getattr(self.solver_config, "t_couple_stepwise_attention_heads", 4)),
                    num_mlp_layers=int(getattr(self.solver_config, "t_couple_stepwise_mlp_layers", 3)),
                    hidden_dim=int(getattr(self.solver_config, "t_couple_stepwise_hidden_dim", 128)),
                    image_encoder_depth=int(getattr(self.solver_config, "t_couple_stepwise_image_encoder_depth", 2)),
                    image_encoder_width=int(getattr(self.solver_config, "t_couple_stepwise_image_encoder_width", 32)),
                    cross_attention_heads=int(getattr(self.solver_config, "t_couple_stepwise_cross_attention_heads", 4)),
                    attention_dim=int(getattr(self.solver_config, "t_couple_stepwise_attention_dim", 128)),
                    number_of_transformer_blocks=int(getattr(self.solver_config, "t_couple_stepwise_transformer_blocks", 3)),
                )
            else:
                raise ValueError(f"Unsupported t_couple_parametrization={self.t_couple_parametrization}")

            self._t_couple_bias = nn.Parameter(torch.zeros(self.steps), requires_grad=self.solver_config.t_couple_requires_grad)
            # placeholder (will be replaced per-batch)
            self.t_couple = torch.zeros(self.steps)

        solver.t_couple = self.t_couple

        # init coef (a/c) parametrizations
        self.a_parametrization = getattr(self.solver_config, "a_parametrization", "diff")
        self.c_parametrization = getattr(self.solver_config, "c_parametrization", "diff")
        self.coef_prediction_mode = getattr(self.solver_config, "coef_prediction_mode", "all_at_once")
        if self.coef_prediction_mode not in ("all_at_once", "step_wise"):
            raise ValueError(
                f"Unsupported coef_prediction_mode={self.coef_prediction_mode}. "
                "Expected one of: all_at_once, step_wise."
            )
        if (
            "stepwise" in {self.a_parametrization, self.c_parametrization, self.t_couple_parametrization}
            and self.coef_prediction_mode != "step_wise"
        ):
            raise ValueError(
                "stepwise parametrizations require coef_prediction_mode=step_wise."
            )
        self.latent_channels = int(getattr(self.solver_config, "latent_channels", 4))
        self.text_embed_dim = int(getattr(self.solver_config, "text_embed_dim", 768))
        self._a_stepwise_table = None
        self._c_stepwise_table = None
        self._t_couple_stepwise_table = None

        self.a_diff_model = None
        self.c_diff_model = None
        self._a_bias = None
        self._c_bias = None

        if self.a_parametrization != "diff":
            out_dim = self.order * self.steps
            if self.a_parametrization == "linear":
                self.a_diff_model = nn.Linear(in_features=768, out_features=out_dim)
                nn.init.normal_(self.a_diff_model.weight, std=0.02)
                nn.init.zeros_(self.a_diff_model.bias)
            elif self.a_parametrization == "mlp":
                self.a_diff_model = nn.Sequential(
                    nn.Linear(in_features=768, out_features=768 * 4, bias=False),
                    nn.LayerNorm(768 * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=768 * 4, out_features=768 * 4, bias=False),
                    nn.LayerNorm(768 * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=768 * 4, out_features=out_dim),
                )
                nn.init.kaiming_uniform_(self.a_diff_model[0].weight, a=0.1)
                nn.init.kaiming_uniform_(self.a_diff_model[3].weight, a=0.1)
                nn.init.xavier_uniform_(self.a_diff_model[6].weight)  # или normal(0, 0.01)
                nn.init.zeros_(self.a_diff_model[6].bias)
            elif self.a_parametrization in ("film_mlp", "prompt_noise_film_mlp"):
                # conditional on both prompt + initial noise statistics
                hidden_dim = int(getattr(self.solver_config, "a_film_hidden_dim", 256))
                self.a_diff_model = PromptNoiseFiLMMlp(out_dim=out_dim, hidden_dim=hidden_dim)
            elif self.a_parametrization == "transformer":
                a_cfg = getattr(config, "a_scheduler_model", None)
                if a_cfg is None:
                    a_cfg = config.scheduler_model
                self.a_diff_model = instantiate(a_cfg, num_timesteps=out_dim)
            elif self.a_parametrization == "stepwise":
                self.a_diff_model = create_stepwise_predictor(
                    out_dim=self.order,
                    version=getattr(self.solver_config, "a_stepwise_version", "medium"),
                    latent_channels=self.latent_channels,
                    text_embed_dim=self.text_embed_dim,
                    encoder_width=int(getattr(self.solver_config, "a_stepwise_encoder_width", 32)),
                    num_conv_blocks=int(getattr(self.solver_config, "a_stepwise_conv_blocks", 2)),
                    attention_heads=int(getattr(self.solver_config, "a_stepwise_attention_heads", 4)),
                    num_mlp_layers=int(getattr(self.solver_config, "a_stepwise_mlp_layers", 3)),
                    hidden_dim=int(getattr(self.solver_config, "a_stepwise_hidden_dim", 256)),
                    image_encoder_depth=int(getattr(self.solver_config, "a_stepwise_image_encoder_depth", 2)),
                    image_encoder_width=int(getattr(self.solver_config, "a_stepwise_image_encoder_width", 32)),
                    cross_attention_heads=int(getattr(self.solver_config, "a_stepwise_cross_attention_heads", 4)),
                    attention_dim=int(getattr(self.solver_config, "a_stepwise_attention_dim", 128)),
                    number_of_transformer_blocks=int(getattr(self.solver_config, "a_stepwise_transformer_blocks", 3)),
                )
            else:
                raise ValueError(f"Unsupported a_parametrization={self.a_parametrization}")
            # used when cond_emb is None
            if self.a_parametrization == "stepwise":
                self._a_bias = nn.Parameter(torch.zeros(self.order), requires_grad=self.solver_config.a_requires_grad)
            else:
                self._a_bias = nn.Parameter(torch.zeros(self.order, self.steps), requires_grad=self.solver_config.a_requires_grad)

        if self.c_parametrization != "diff":
            out_dim = self.order * self.steps
            if self.c_parametrization == "linear":
                self.c_diff_model = nn.Linear(in_features=768, out_features=out_dim)
                nn.init.normal_(self.c_diff_model.weight, std=0.02)
                nn.init.zeros_(self.c_diff_model.bias)
            elif self.c_parametrization == "mlp":
                self.c_diff_model = nn.Sequential(
                    nn.Linear(in_features=768, out_features=768 * 4, bias=False),
                    nn.LayerNorm(768 * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=768 * 4, out_features=768 * 4, bias=False),
                    nn.LayerNorm(768 * 4),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=768 * 4, out_features=out_dim),
                )
                nn.init.kaiming_uniform_(self.c_diff_model[0].weight, a=0.2)
                nn.init.kaiming_uniform_(self.c_diff_model[3].weight, a=0.2)
                nn.init.zeros_(self.c_diff_model[6].weight)
                nn.init.zeros_(self.c_diff_model[6].bias)
            elif self.c_parametrization in ("film_mlp", "prompt_noise_film_mlp"):
                # conditional on both prompt + initial noise statistics
                hidden_dim = int(getattr(self.solver_config, "c_film_hidden_dim", 256))
                self.c_diff_model = PromptNoiseFiLMMlp(out_dim=out_dim, hidden_dim=hidden_dim)
            elif self.c_parametrization == "transformer":
                c_cfg = getattr(config, "c_scheduler_model", None)
                if c_cfg is None:
                    c_cfg = config.scheduler_model
                self.c_diff_model = instantiate(c_cfg, num_timesteps=out_dim)
            elif self.c_parametrization == "stepwise":
                self.c_diff_model = create_stepwise_predictor(
                    out_dim=self.order,
                    version=getattr(self.solver_config, "c_stepwise_version", "medium"),
                    latent_channels=self.latent_channels,
                    text_embed_dim=self.text_embed_dim,
                    encoder_width=int(getattr(self.solver_config, "c_stepwise_encoder_width", 32)),
                    num_conv_blocks=int(getattr(self.solver_config, "c_stepwise_conv_blocks", 2)),
                    attention_heads=int(getattr(self.solver_config, "c_stepwise_attention_heads", 4)),
                    num_mlp_layers=int(getattr(self.solver_config, "c_stepwise_mlp_layers", 3)),
                    hidden_dim=int(getattr(self.solver_config, "c_stepwise_hidden_dim", 256)),
                    image_encoder_depth=int(getattr(self.solver_config, "c_stepwise_image_encoder_depth", 2)),
                    image_encoder_width=int(getattr(self.solver_config, "c_stepwise_image_encoder_width", 32)),
                    cross_attention_heads=int(getattr(self.solver_config, "c_stepwise_cross_attention_heads", 4)),
                    attention_dim=int(getattr(self.solver_config, "c_stepwise_attention_dim", 128)),
                    number_of_transformer_blocks=int(getattr(self.solver_config, "c_stepwise_transformer_blocks", 3)),
                )
            else:
                raise ValueError(f"Unsupported c_parametrization={self.c_parametrization}")
            if self.c_parametrization == "stepwise":
                self._c_bias = nn.Parameter(torch.zeros(self.order), requires_grad=self.solver_config.c_requires_grad)
            else:
                self._c_bias = nn.Parameter(torch.zeros(self.order, self.steps), requires_grad=self.solver_config.c_requires_grad)

        # baseline coefficients (current behavior)
        # ====== A coefficients ======
        if self.a_parametrization == "diff":
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
        if self.c_parametrization == "diff":
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
        if self.a_parametrization == "diff":
            # diff params already created with requires_grad flag
            pass
        else:
            # control conditional model
            set_requires_grad(self.a_diff_model, self.solver_config.a_requires_grad)
            if self._a_bias is not None:
                self._a_bias.requires_grad = self.solver_config.a_requires_grad


        # ---- C parameters ----
        if self.c_parametrization == "diff":
            pass
        else:
            set_requires_grad(self.c_diff_model, self.solver_config.c_requires_grad)
            if self._c_bias is not None:
                self._c_bias.requires_grad = self.solver_config.c_requires_grad

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

        self.coeff_residual_scale = 0.01  # маленькая амплитуда

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
            zero_last_layer(self.a_diff_model)

        # C model
        if self.c_diff_model is not None:
            zero_last_layer(self.c_diff_model)

        # Расписание t: те же начальные логиты, что у linear/mlp (inv stick-breaking для равномерной сетки)
        if self.t_parametrization == "transformer":
            t_unif = torch.linspace(1.0, self.t_eps, self.steps + 1).flip(0)
            inv_logits = self.get_inv_t_steps(t_unif)
            _init_scheduler_transformer_residual_head(self.mu_logit, logistic_bias_1d=inv_logits)
        
        # end init solver
        self.solver = solver

    def _update_dynamic_t_couple(
        self,
        noise: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
        step_idx: Optional[int] = None,
    ) -> None:
        if self.t_couple_parametrization == "diff":
            return

        if self.t_couple_parametrization == "stepwise":
            b = noise.shape[0]
            if (
                self._t_couple_stepwise_table is None
                or self._t_couple_stepwise_table.shape != (b, self.steps)
                or self._t_couple_stepwise_table.device != noise.device
                or self._t_couple_stepwise_table.dtype != noise.dtype
            ):
                self._t_couple_stepwise_table = torch.zeros(
                    b, self.steps, device=noise.device, dtype=noise.dtype
                )
            if cond_emb is not None:
                t_couple_cur = self.t_couple_model(noise, cond_emb).reshape(b)
            else:
                t_couple_cur = self._t_couple_bias.reshape(1).expand(b).to(
                    device=noise.device, dtype=noise.dtype
                )
            if step_idx is None:
                self._t_couple_stepwise_table[:] = t_couple_cur.unsqueeze(1)
            else:
                self._t_couple_stepwise_table[:, step_idx] = t_couple_cur
            self.solver.t_couple = self._t_couple_stepwise_table
            return

        if self.t_couple_parametrization in ("linear", "mlp"):
            if cond_emb is not None:
                x = cond_emb.mean(dim=1)
                t_couple = self.t_couple_model(x)  # (B, steps)
            else:
                t_couple = self._t_couple_bias.reshape(1, self.steps)
        else:  # film_mlp | prompt_noise_film_mlp
            if cond_emb is not None:
                t_couple = self.t_couple_model(noise, cond_emb)  # (B, steps)
            else:
                t_couple = self._t_couple_bias.reshape(1, self.steps)

        self.solver.t_couple = t_couple

    def _update_dynamic_ac_coefs(
        self,
        noise: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
        step_idx: Optional[int] = None,
    ) -> None:
        """
        If enabled by config, compute batch-dependent a/c coefficient tables and
        attach them to the solver as tensors of shape (B, steps).
        """
        if self.a_parametrization != "diff":
            out_dim = self.order * self.steps
            if self.a_parametrization in ("linear", "mlp"):
                if cond_emb is not None:
                    x = cond_emb.mean(dim=1)
                    a_flat = self.a_diff_model(x)  # (B, out_dim)
                else:
                    a_flat = self._a_bias.reshape(1, out_dim)
            elif self.a_parametrization in ("film_mlp", "prompt_noise_film_mlp"):
                if cond_emb is not None:
                    a_flat = self.a_diff_model(noise, cond_emb)  # (B, out_dim)
                else:
                    a_flat = self._a_bias.reshape(1, out_dim)
            elif self.a_parametrization == "stepwise":
                a_flat = None
            else:  # transformer
                if cond_emb is not None:
                    a_flat = self.a_diff_model(noise, cond_emb)  # (B, out_dim)
                else:
                    dev = noise.device
                    dummy_cond = torch.zeros(1, 77, 768, device=dev)
                    a_flat = self.a_diff_model(noise, dummy_cond)  # (1, out_dim)
            if self.a_parametrization == "stepwise":
                b = noise.shape[0]
                if (
                    self._a_stepwise_table is None
                    or self._a_stepwise_table.shape != (b, self.order, self.steps)
                    or self._a_stepwise_table.device != noise.device
                    or self._a_stepwise_table.dtype != noise.dtype
                ):
                    self._a_stepwise_table = torch.zeros(
                        b, self.order, self.steps, device=noise.device, dtype=noise.dtype
                    )
                if cond_emb is not None:
                    a_cur = self.a_diff_model(noise, cond_emb)  # (B, order)
                else:
                    a_cur = self._a_bias.reshape(1, self.order).expand(b, self.order).to(
                        device=noise.device, dtype=noise.dtype
                    )
                if step_idx is None:
                    self._a_stepwise_table[:] = a_cur.unsqueeze(-1)
                else:
                    self._a_stepwise_table[:, :, step_idx] = a_cur
                a_all = self._a_stepwise_table
            else:
                a_all = a_flat.reshape(a_flat.shape[0], self.order, self.steps)
            
            for i in range(1, self.order + 1):
                self.solver.__setattr__(f"a{i}_diff", a_all[:, i - 1, :])

        if self.c_parametrization != "diff":
            out_dim = self.order * self.steps
            if self.c_parametrization in ("linear", "mlp"):
                if cond_emb is not None:
                    x = cond_emb.mean(dim=1)
                    c_flat = self.c_diff_model(x)  # (B, out_dim)
                else:
                    c_flat = self._c_bias.reshape(1, out_dim)
            elif self.c_parametrization in ("film_mlp", "prompt_noise_film_mlp"):
                if cond_emb is not None:
                    c_flat = self.c_diff_model(noise, cond_emb)  # (B, out_dim)
                else:
                    c_flat = self._c_bias.reshape(1, out_dim)
            elif self.c_parametrization == "stepwise":
                c_flat = None
            else:  # transformer
                if cond_emb is not None:
                    c_flat = self.c_diff_model(noise, cond_emb)  # (B, out_dim)
                else:
                    dev = noise.device
                    dummy_cond = torch.zeros(1, 77, 768, device=dev)
                    c_flat = self.c_diff_model(noise, dummy_cond)  # (1, out_dim)
            if self.c_parametrization == "stepwise":
                b = noise.shape[0]
                if (
                    self._c_stepwise_table is None
                    or self._c_stepwise_table.shape != (b, self.order, self.steps)
                    or self._c_stepwise_table.device != noise.device
                    or self._c_stepwise_table.dtype != noise.dtype
                ):
                    self._c_stepwise_table = torch.zeros(
                        b, self.order, self.steps, device=noise.device, dtype=noise.dtype
                    )
                if cond_emb is not None:
                    c_cur = self.c_diff_model(noise, cond_emb)  # (B, order)
                else:
                    c_cur = self._c_bias.reshape(1, self.order).expand(b, self.order).to(
                        device=noise.device, dtype=noise.dtype
                    )
                if step_idx is None:
                    self._c_stepwise_table[:] = c_cur.unsqueeze(-1)
                else:
                    self._c_stepwise_table[:, :, step_idx] = c_cur
                c_all = self._c_stepwise_table
            else:
                c_all = c_flat.reshape(c_flat.shape[0], self.order, self.steps)
            
            for i in range(1, self.order + 1):
                self.solver.__setattr__(f"c{i}_diff", c_all[:, i - 1, :])
            
    # timesteps logic
    def get_t_steps(self, noise=None, cond_emb=None, **kwargs) -> torch.Tensor:
        """Get generation timesteps."""
        if self.t_parametrization in ("diff", "mu_logit"):
            logits = self.mu_logit
        elif self.t_parametrization == "linear":
            if cond_emb is not None:
                # B = cond_emb.shape[0]
                if torch.isnan(self.mu_logit.weight).any():
                    print(f"⚠️ NaN в mu_logit.weight!")
                if torch.isnan(self.mu_logit.bias).any():
                    print(f"⚠️ NaN в mu_logit.bias!")
                cond_emb = cond_emb.mean(dim=1)
		        # cond_emb = cond_emb / cond_emb.norm(dim=-1, keepdim=True)
                logits = self.mu_logit(cond_emb)
		        # logits = self.mu_logit(cond_emb.reshape(B, -1)).T
            else:
                logits = self.mu_logit.bias
        elif self.t_parametrization == "mlp":
            if cond_emb is not None:
                logits = self.mu_logit(cond_emb)[:, 0, :]
            else:
                logits = self.mu_logit[2].bias
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

    def student_sampler_fn(self, noise: torch.Tensor, **kwargs) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Calls `sample` method of the Generalised Solver. 
        
        Args:
            noise (torch.Tensor): An initial noise tensor to start sampling process from.
        
        Returns:
            None: A placeholder for consistency with latent models.
            torch.tensor: Sampled images
        """
        cond_emb = getattr(self.model.model_fn, "condition", None)
        if self.coef_prediction_mode == "all_at_once":
            self._update_dynamic_t_couple(noise=noise, cond_emb=cond_emb)
            self._update_dynamic_ac_coefs(noise=noise, cond_emb=cond_emb)

        before_step_fn = None
        if self.coef_prediction_mode == "step_wise":
            def before_step_fn(step_idx: int, x: torch.Tensor) -> None:
                self._update_dynamic_t_couple(noise=x, cond_emb=cond_emb, step_idx=step_idx)
                self._update_dynamic_ac_coefs(noise=x, cond_emb=cond_emb, step_idx=step_idx)

        images = self.solver.sample(
            x=noise,
            steps=self.steps,
            order=self.order,
            before_step_fn=before_step_fn,
        )
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
        assert len(batch) == 4, f"len(batch) is expected to be 4, yours is {len(batch)}"
        noise, images, _, _ = batch

        d = {}
        if return_timesteps:
            # For conditional t parametrizations (e.g. film_mlp), timesteps depend on cond_emb (+ noise),
            # so pass them explicitly (same behavior as in GSWrapperLatent).
            cond_emb = getattr(self.model.model_fn, "condition", None)
            d['timesteps'] = self.solver.get_time_steps(noise, cond_emb)
        _, student_images = self.student_sampler_fn(noise)

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
        if self.coef_prediction_mode == "all_at_once":
            self._update_dynamic_t_couple(noise=noise, cond_emb=cond_emb)
            self._update_dynamic_ac_coefs(noise=noise, cond_emb=cond_emb)

        before_step_fn = None
        if self.coef_prediction_mode == "step_wise":
            def before_step_fn(step_idx: int, x: torch.Tensor) -> None:
                self._update_dynamic_t_couple(noise=x, cond_emb=cond_emb, step_idx=step_idx)
                self._update_dynamic_ac_coefs(noise=x, cond_emb=cond_emb, step_idx=step_idx)

        latents = self.solver.sample(
            x=noise,
            steps=self.steps,
            order=self.order,
            before_step_fn=before_step_fn,
        )

        if decode:
            images = self.model.decode(latents)

        return latents, images
    
    def forward(self, batch: SyntDataType, return_timesteps: bool = False, is_train: bool = True) -> dict:
        assert len(batch) == 4, f"len(batch) is expected to be 4, yours is {len(batch)}"
        noise, images, latents, condition = batch

        d = {}
        if return_timesteps:
            if condition is not None:
                self.model.set_condition(condition)
            cond_emb = self.model.model_fn.condition
            print('??', cond_emb.shape if cond_emb is not None else None)
            d['timesteps'] = self.solver.get_time_steps(noise, cond_emb)
            print(d['timesteps'])
            print(condition)
            # print(cond_emb)
        student_latents, _ = self.student_sampler_fn(
            noise,
            condition=condition
        )

        d['loss_l1_latents'] = torch.abs(latents - student_latents).mean((1, 2, 3))
        d['loss_l2_latents'] = torch.square(latents - student_latents).mean((1, 2, 3))
        d['x0_t'] = self.interpolate_lpips(images)
        d['latents_s'] = student_latents

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

        d['loss_total'] = self.loss_config.disc_weight * d.get('gen_loss_adv', 0.) + d[self.loss_config.loss_key]

        return d
