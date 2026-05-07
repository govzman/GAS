from typing import Dict

import torch

from src.model.gas.solver_param_predictor import SolverParamPredictor


def check_solver_param_predictor_shapes(
    batch_size: int,
    steps: int,
    order: int,
    feature_dim: int = 256,
) -> Dict[str, torch.Size]:
    predictor = SolverParamPredictor(
        steps=steps,
        order=order,
        feature_dim=feature_dim,
        hidden_dim=128,
    )
    feats = torch.randn(batch_size, feature_dim)
    out = predictor(feats)
    expected = {}
    for i in range(1, order + 1):
        expected[f"a{i}_diff"] = torch.Size([batch_size, steps])
        expected[f"c{i}_diff"] = torch.Size([batch_size, steps])
    expected["t_couple"] = torch.Size([batch_size, steps])
    for k, shp in expected.items():
        assert k in out, f"Missing key {k}"
        assert out[k].shape == shp, f"{k} shape mismatch: {out[k].shape} vs {shp}"
    return {k: v.shape for k, v in out.items()}
