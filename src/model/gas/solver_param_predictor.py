from typing import Dict, Optional

import torch
from torch import nn


class SolverParamPredictor(nn.Module):
    """
    Optional hypernetwork for per-sample GS residual parameters:
    - phi (a/c diffs)
    - xi (t_couple)
    """

    def __init__(
        self,
        steps: int,
        order: int,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        predict_a_diff: bool = True,
        predict_c_diff: bool = True,
        predict_t_couple: bool = True,
        clamp_enabled: bool = True,
        clamp_max_abs: float = 0.1,
        stochastic_enabled: bool = False,
        stochastic_fixed_std: float = 0.01,
    ):
        super().__init__()
        self.steps = int(steps)
        self.order = int(order)
        self.predict_a_diff = bool(predict_a_diff)
        self.predict_c_diff = bool(predict_c_diff)
        self.predict_t_couple = bool(predict_t_couple)
        self.clamp_enabled = bool(clamp_enabled)
        self.clamp_max_abs = float(clamp_max_abs)
        self.stochastic_enabled = bool(stochastic_enabled)
        self.stochastic_fixed_std = float(stochastic_fixed_std)

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.heads = nn.ModuleDict()
        if self.predict_a_diff:
            self.heads["a"] = nn.Linear(hidden_dim, self.order * self.steps)
        if self.predict_c_diff:
            self.heads["c"] = nn.Linear(hidden_dim, self.order * self.steps)
        if self.predict_t_couple:
            self.heads["t_couple"] = nn.Linear(hidden_dim, self.steps)
        self._init_zero()

    def _init_zero(self) -> None:
        for mod in self.heads.values():
            nn.init.zeros_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    @staticmethod
    def features_from_latent_and_condition(
        x: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
        feature_dim: int,
    ) -> torch.Tensor:
        b = x.shape[0]
        x_feat = x.reshape(b, -1).mean(dim=1, keepdim=True)
        x_std = x.reshape(b, -1).std(dim=1, keepdim=True, unbiased=False)
        x_rms = (x.reshape(b, -1).pow(2).mean(dim=1, keepdim=True) + 1e-12).sqrt()
        base = torch.cat([x_feat, x_std, x_rms], dim=1)
        if cond_emb is not None:
            c = cond_emb.mean(dim=1)
            c_mean = c.mean(dim=1, keepdim=True)
            c_std = c.std(dim=1, keepdim=True, unbiased=False)
            base = torch.cat([base, c_mean, c_std], dim=1)
        if base.shape[1] < feature_dim:
            pad = torch.zeros(b, feature_dim - base.shape[1], device=x.device, dtype=x.dtype)
            base = torch.cat([base, pad], dim=1)
        return base[:, :feature_dim]

    def _clamp(self, t: torch.Tensor) -> torch.Tensor:
        if self.clamp_enabled:
            return t.clamp(-self.clamp_max_abs, self.clamp_max_abs)
        return t

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(features)
        out: Dict[str, torch.Tensor] = {}
        if "a" in self.heads:
            a = self.heads["a"](h).reshape(features.shape[0], self.order, self.steps)
            for i in range(self.order):
                out[f"a{i+1}_diff"] = self._clamp(a[:, i, :])
        if "c" in self.heads:
            c = self.heads["c"](h).reshape(features.shape[0], self.order, self.steps)
            for i in range(self.order):
                out[f"c{i+1}_diff"] = self._clamp(c[:, i, :])
        if "t_couple" in self.heads:
            out["t_couple"] = self._clamp(self.heads["t_couple"](h))
        return out

    def sample_with_logprob(self, features: torch.Tensor):
        mean_out = self.forward(features)
        if not self.stochastic_enabled:
            return mean_out, None, None
        log_prob = None
        entropy = None
        sampled: Dict[str, torch.Tensor] = {}
        var = self.stochastic_fixed_std ** 2
        const = -0.5 * torch.log(torch.tensor(2.0 * torch.pi * var, device=features.device, dtype=features.dtype))
        for k, m in mean_out.items():
            eps = torch.randn_like(m)
            s = m + self.stochastic_fixed_std * eps
            sampled[k] = s
            lp = const - 0.5 * ((s - m) ** 2) / var
            cur_lp = lp.flatten(start_dim=1).sum(dim=1)
            cur_ent = (-lp).flatten(start_dim=1).mean(dim=1)
            log_prob = cur_lp if log_prob is None else (log_prob + cur_lp)
            entropy = cur_ent if entropy is None else (entropy + cur_ent)
        return sampled, log_prob, entropy
