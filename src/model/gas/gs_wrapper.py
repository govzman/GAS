import contextlib
import lpips

import numpy as np
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
from src.scheduler_model.film_mlp import PromptNoiseFiLMMlp


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
        
        self._retain_solver_coeff_grads: bool = False

        # setup solver
        solver = self.get_base_solver()
        self.steps = self.solver_config.steps
        self.order = self.solver_config.order
        self.latent_channels = int(getattr(self.solver_config, "latent_channels", 4))
        self.text_embed_dim = int(getattr(self.solver_config, "text_embed_dim", 768))

        # init t steps
        self.eps_mu_offset = 1e-5
        # support both new and legacy config key names
        self.t_parametrization = getattr(
            self.solver_config,
            "t_parametrization",
            getattr(self.solver_config, "t_schedule_parametrization", "diff"),
        )

        # Timestep logits: static diff, FiLM(noise, prompt), or transformer.
        if self.t_parametrization in ("diff", "mu_logit"):
            self.mu_logit = nn.Parameter(torch.ones(self.steps - 1), requires_grad=self.solver_config.t_requires_grad)
            t_unif = torch.linspace(1., self.t_eps, self.steps + 1).flip(0)
            self.mu_logit.data = self.get_inv_t_steps(t_unif)
        elif self.t_parametrization == "film_mlp":
            hidden_dim = int(getattr(self.solver_config, "t_film_hidden_dim", 256))
            self.mu_logit = PromptNoiseFiLMMlp(
                out_dim=self.steps - 1,
                hidden_dim=hidden_dim,
                in_channels=self.latent_channels,
            )
        elif self.t_parametrization == "transformer":
            self.mu_logit = self._instantiate_with_latent_channels(config.scheduler_model)
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
            if self.t_couple_parametrization == "film_mlp":
                hidden_dim = int(getattr(self.solver_config, "t_couple_film_hidden_dim", 256))
                self.t_couple_model = PromptNoiseFiLMMlp(
                    out_dim=self.steps,
                    hidden_dim=hidden_dim,
                    in_channels=self.latent_channels,
                )
            else:
                raise ValueError(f"Unsupported t_couple_parametrization={self.t_couple_parametrization}")

            self._t_couple_bias = nn.Parameter(torch.zeros(self.steps), requires_grad=self.solver_config.t_couple_requires_grad)
            # placeholder (will be replaced per-batch)
            self.t_couple = torch.zeros(self.steps)

        solver.t_couple = self.t_couple

        # a/c: one network head predicts concatenated [a_flat | c_flat] (2 * order * steps),
        # then slices become per-order coefficient tables (absolute prediction).
        # Modes: diff | film_mlp | transformer
        self.a_parametrization = getattr(self.solver_config, "a_parametrization", "diff")
        self.c_parametrization = getattr(self.solver_config, "c_parametrization", "diff")
        self._validate_ac_parametrization(self.a_parametrization, "a_parametrization")
        self._validate_ac_parametrization(self.c_parametrization, "c_parametrization")
        self.ac_coeff_model = None
        self._ac_bias = None
        self._ac_half_dim = self.order * self.steps

        self._init_ac_coeff_model(config)

        # Learnable global coeffs only for pure diff mode.
        if self.a_parametrization == "diff":
            for i in range(1, self.order + 1):
                aname = f"a{i}_diff"
                self.register_parameter(
                    name=aname,
                    param=nn.Parameter(
                        torch.zeros(self.steps),
                        requires_grad=self.solver_config.a_requires_grad,
                    ),
                )
                solver.__setattr__(aname, self.__getattr__(aname))
        else:
            for i in range(1, self.order + 1):
                solver.__setattr__(f"a{i}_diff", torch.zeros(self.steps))

        if self.c_parametrization == "diff":
            for i in range(1, self.order + 1):
                cname = f"c{i}_diff"
                self.register_parameter(
                    name=cname,
                    param=nn.Parameter(
                        torch.zeros(self.steps),
                        requires_grad=self.solver_config.c_requires_grad,
                    ),
                )
                solver.__setattr__(cname, self.__getattr__(cname))
        else:
            for i in range(1, self.order + 1):
                solver.__setattr__(f"c{i}_diff", torch.zeros(self.steps))

        # theory coef
        solver.use_theory_coef = self.solver_config.use_theory_coef
        if not solver.use_theory_coef:
            solver.init_coefs(
                steps=self.steps,
                order=self.order,
                timesteps=self.get_t_steps(),
            )

        def set_requires_grad(module, flag: bool):
            if module is None:
                return
            for p in module.parameters():
                p.requires_grad = flag

        if self.ac_coeff_model is not None:
            ac_grad = bool(getattr(self.solver_config, "a_requires_grad", True)) or bool(
                getattr(self.solver_config, "c_requires_grad", True)
            )
            set_requires_grad(self.ac_coeff_model, ac_grad)
            if self._ac_bias is not None:
                self._ac_bias.requires_grad = ac_grad

        if self.t_couple_parametrization != "diff":
            set_requires_grad(self.t_couple_model, self.solver_config.t_couple_requires_grad)
            if self._t_couple_bias is not None:
                self._t_couple_bias.requires_grad = self.solver_config.t_couple_requires_grad

        self.solver = solver

    @staticmethod
    def _validate_ac_parametrization(mode: str, name: str) -> None:
        allowed = {"diff", "film_mlp", "transformer"}
        if mode not in allowed:
            raise ValueError(
                f"Unsupported {name}={mode}. Allowed: {sorted(allowed)}"
            )

    @staticmethod
    def _ac_mode_needs_network(mode: str) -> bool:
        return mode not in ("diff",)

    @staticmethod
    def _ac_network_family(mode: str) -> Optional[str]:
        if mode == "film_mlp":
            return "film"
        if mode == "transformer":
            return "transformer"
        return None

    def _init_ac_coeff_model(self, config: ConfigDict) -> None:
        """Build one FiLM/transformer head that outputs [a_flat | c_flat]."""
        a_needs = self._ac_mode_needs_network(self.a_parametrization)
        c_needs = self._ac_mode_needs_network(self.c_parametrization)
        if not a_needs and not c_needs:
            return

        families = {
            f
            for f in (
                self._ac_network_family(self.a_parametrization),
                self._ac_network_family(self.c_parametrization),
            )
            if f is not None
        }
        if len(families) > 1:
            raise ValueError(
                f"a_parametrization={self.a_parametrization} and "
                f"c_parametrization={self.c_parametrization} need different network "
                f"families {families}; use one head type for both."
            )
        family = next(iter(families))
        out_dim = 2 * self._ac_half_dim

        if family == "film":
            hidden_dim = int(
                getattr(
                    self.solver_config,
                    "ac_film_hidden_dim",
                    getattr(
                        self.solver_config,
                        "a_film_hidden_dim",
                        getattr(self.solver_config, "c_film_hidden_dim", 256),
                    ),
                )
            )
            self.ac_coeff_model = PromptNoiseFiLMMlp(
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                in_channels=self.latent_channels,
            )
        else:
            ac_cfg = getattr(config, "ac_scheduler_model", None)
            if ac_cfg is None:
                ac_cfg = getattr(config, "a_scheduler_model", None)
            if ac_cfg is None:
                ac_cfg = getattr(config, "c_scheduler_model", None)
            if ac_cfg is None:
                ac_cfg = config.scheduler_model
            self.ac_coeff_model = self._instantiate_with_latent_channels(
                ac_cfg, num_timesteps=out_dim
            )

        self._ac_bias = nn.Parameter(torch.zeros(out_dim), requires_grad=True)

    def _instantiate_with_latent_channels(self, cfg, **kwargs):
        """Hydra-instantiate a scheduler model.

        FiLM configs need ``in_channels`` aligned with latents; transformer uses its
        own ImageEncoder and must not receive that kwarg.
        """
        from omegaconf import OmegaConf

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg_dict, dict):
            cfg_dict = dict(cfg_dict)
            target = str(cfg_dict.get("_target_", ""))
            if "PromptNoiseFiLMMlp" in target:
                cfg_dict["in_channels"] = self.latent_channels
            else:
                cfg_dict.pop("in_channels", None)
            cfg = OmegaConf.create(cfg_dict)
        return instantiate(cfg, **kwargs)

    def should_log_final_solver_coeffs(self) -> bool:
        """True when effective coeffs live on the solver (not only static diff Parameters)."""
        if self.a_parametrization != "diff" or self.c_parametrization != "diff":
            return True
        if self.t_couple_parametrization != "diff":
            return True
        return False

    def set_retain_solver_coeff_grads(self, enabled: bool) -> None:
        self._retain_solver_coeff_grads = bool(enabled)

    def _retain_solver_coeff_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._retain_solver_coeff_grads and tensor.requires_grad:
            tensor.retain_grad()
        return tensor

    def _attach_solver_coeff(self, name: str, tensor: torch.Tensor) -> None:
        self.solver.__setattr__(name, self._retain_solver_coeff_if_needed(tensor))

    def get_final_solver_coeff_tensors(self) -> Dict[str, torch.Tensor]:
        """Coefficients currently on the solver after sampling (may be batch-dependent)."""
        out: Dict[str, torch.Tensor] = {}
        for i in range(1, self.order + 1):
            for prefix in ("a", "c"):
                name = f"{prefix}{i}_diff"
                t = getattr(self.solver, name, None)
                if isinstance(t, torch.Tensor):
                    out[name] = t
        t_couple = getattr(self.solver, "t_couple", None)
        if isinstance(t_couple, torch.Tensor):
            out["t_couple"] = t_couple
        return out

    @staticmethod
    def _reduce_coeff_tensor_for_logging(
        tensor: torch.Tensor,
        reduction: str = "mean",
    ) -> np.ndarray:
        t = tensor.detach().float()
        if t.ndim == 0:
            return t.cpu().numpy().reshape(1)
        if t.ndim == 1:
            return t.cpu().numpy()
        if reduction == "batch0":
            return t[0].cpu().numpy()
        return t.mean(dim=0).cpu().numpy()

    def get_final_solver_coeffs_for_logging(
        self,
        reduction: str = "mean",
    ) -> Dict[str, np.ndarray]:
        return {
            name: self._reduce_coeff_tensor_for_logging(tensor, reduction=reduction)
            for name, tensor in self.get_final_solver_coeff_tensors().items()
        }

    def get_final_solver_coeff_grads_for_logging(
        self,
        reduction: str = "mean",
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, tensor in self.get_final_solver_coeff_tensors().items():
            if tensor.grad is not None:
                out[name] = self._reduce_coeff_tensor_for_logging(tensor.grad, reduction=reduction)
        return out

    def _update_dynamic_t_couple(
        self,
        noise: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> None:
        if self.t_couple_parametrization == "diff":
            return

        if cond_emb is not None:
            t_couple = self.t_couple_model(noise, cond_emb)
        else:
            t_couple = self._t_couple_bias.reshape(1, self.steps)
        self._attach_solver_coeff("t_couple", t_couple)

    def _predict_ac_flat(
        self,
        noise: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the single a/c head and split into a_flat, c_flat (each order*steps)."""
        half = self._ac_half_dim
        if cond_emb is not None:
            flat = self.ac_coeff_model(noise, cond_emb)
        else:
            flat = self._ac_bias.reshape(1, -1).expand(noise.shape[0], -1)
        return flat[:, :half], flat[:, half:]

    def _attach_ac_side(
        self,
        prefix: str,
        parametrization: str,
        pred_flat: torch.Tensor,
    ) -> None:
        """Attach a* or c* tables from a flat (B, order*steps) prediction."""
        if parametrization == "diff":
            return

        tables = pred_flat.reshape(pred_flat.shape[0], self.order, self.steps)
        for i in range(1, self.order + 1):
            self._attach_solver_coeff(f"{prefix}{i}_diff", tables[:, i - 1, :])

    def _update_dynamic_ac_coefs(
        self,
        noise: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> None:
        """
        Predict batch-dependent a/c coefficient tables with one head and attach
        them to the solver as tensors of shape (B, steps).

        Output layout: [a_flat | c_flat], each of length order * steps.
        Pure ``diff`` sides keep their registered parameters unchanged.
        """
        if self.ac_coeff_model is None:
            return

        a_flat, c_flat = self._predict_ac_flat(noise, cond_emb)
        self._attach_ac_side("a", self.a_parametrization, a_flat)
        self._attach_ac_side("c", self.c_parametrization, c_flat)

    # timesteps logic
    def get_t_steps(self, noise=None, cond_emb=None, **kwargs) -> torch.Tensor:
        """Get generation timesteps."""
        if self.t_parametrization in ("diff", "mu_logit"):
            logits = self.mu_logit
        elif self.t_parametrization == "film_mlp":
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
        """Per-batch-element MSE/MAE vs teacher ``manual_solver_params`` when present.

        Used on the synthetic dataset path where batch[4] carries teacher GT coeffs.
        """
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
        """Temporarily override solver coeffs/timesteps with teacher GT (synthetic data)."""
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
            if manual_solver_params is None:
                self._update_dynamic_t_couple(noise=noise, cond_emb=cond_emb)
                self._update_dynamic_ac_coefs(noise=noise, cond_emb=cond_emb)

            images = self.solver.sample(
                x=noise,
                steps=self.steps,
                order=self.order,
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
            if manual_solver_params is None:
                self._update_dynamic_t_couple(noise=noise, cond_emb=cond_emb)
                self._update_dynamic_ac_coefs(noise=noise, cond_emb=cond_emb)

            latents = self.solver.sample(
                x=noise,
                steps=self.steps,
                order=self.order,
            )

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
        d["loss_total"] = adv_part + base_loss

        return d
