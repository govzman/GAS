import torch
import pickle
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Optional, Union, List, Tuple, Sequence

from omegaconf import DictConfig

SyntDataType = Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[Union[torch.Tensor, List[str]]],
    Optional[Dict[str, torch.Tensor]],
]

GT_SOLVER_PREFIX = "manual_solver_params."


def move_batch_to_device(batch: Tuple[Any, ...], device: torch.device) -> Tuple[Any, ...]:
    """Move tensors in batch to device; supports optional dict of GT solver tensors."""
    out: List[Any] = []
    for v in batch:
        if isinstance(v, torch.Tensor):
            out.append(v.to(device))
        elif isinstance(v, dict):
            out.append(
                {
                    k: t.to(device) if isinstance(t, torch.Tensor) else t
                    for k, t in v.items()
                }
            )
        else:
            out.append(v)
    return tuple(out)


class SyntDataset(Dataset):
    """Dataset class.
    Expects dataset in format as done in generate.py / collate.py (teacher pickle).

    If the pickle contains flattened keys ``manual_solver_params.<name>``, each sample
    returns a dict of those tensors as the 5th tuple element for training-time GT comparison.
    """

    def __init__(self, dataset_path: str):
        with open(dataset_path, "rb") as fp:
            self.data = pickle.load(fp)

        self.noise_key = "noise"
        self.images_key = "images"
        self.latent_key = "latents"
        self.condition_key = "condition"

        self.gt_solver_param_names: List[str] = sorted(
            k[len(GT_SOLVER_PREFIX) :]
            for k in self.data.keys()
            if k.startswith(GT_SOLVER_PREFIX)
        )

    def __len__(self):
        return len(self.data[self.images_key])

    def __getitem__(self, idx):
        noise = self.data[self.noise_key][idx]
        images = self.data[self.images_key][idx]
        latents = self.data[self.latent_key][idx]
        condition = self.data[self.condition_key][idx]

        gt_solver_params = None
        if self.gt_solver_param_names:
            gt_solver_params = {
                name: self.data[f"{GT_SOLVER_PREFIX}{name}"][idx]
                for name in self.gt_solver_param_names
            }

        return noise, images, latents, condition, gt_solver_params


class SyntDataLoaders:
    """Synthetic dataset loaders class.
    
    Class contatining all required dataloaders for GS/GAS training: 
    train and test loaders, batch for visulization.  
    Does not shuffle the dataset for reproducibility. 
    
    Attributes:
        train_loader (DataLoader): Dataloader with train data subset.
            Contains first `config.train_size` items from the whole dataset (teacher pickle file).
        test_loader (DataLoader): Dataloader with test data subset.
            Contains first `config.validation_size` items from the whole dataset (teacher pickle file).
        vis_batch (tuple): The first batch of the train subset for logging visualization purposes.
    """

    def __init__(self, config: DictConfig):
        self.config = config

        dataset = SyntDataset(dataset_path=self.config.teacher_pkl)

        assert len(dataset) >= self.config.train_size + self.config.validation_size, f"""
            You'll have train data in validation split:
            your train_size={self.config.train_size}, val_size={self.config.validation_size},
            while the dataset size is {len(dataset)}
        """

        train_dataset = torch.utils.data.Subset(
            dataset, range(self.config.dataset_shift, self.config.dataset_shift + self.config.train_size))

        test_dataset = torch.utils.data.Subset(
            dataset, range(len(dataset) - self.config.validation_size, len(dataset)))
            # dataset, range(self.config.dataset_shift, self.config.dataset_shift + self.config.validation_size))

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.validation_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn
        )

        self.vis_batch = next(
            iter(
                DataLoader(
                    train_dataset,
                    batch_size=self.config.size_vis,
                    shuffle=False,
                    collate_fn=self.collate_fn
                )
            )
        )

        n_gt = len(dataset.gt_solver_param_names)
        print(f"""
            -------------- Dataloader info --------------
            \tUse latents = {self.config.use_latents}
            \tUse condition = {self.config.use_condition}
            \tGT solver params in teacher pickle = {n_gt} tensors
            \tlen(train_loader) = {len(self.train_loader)}
            \tlen(test_loader) = {len(self.test_loader)}
        """)

    def collate_fn(self, batch: Sequence[SyntDataType]) -> SyntDataType:
        """Collates synthetic dataset from teacher pickle into batch.

        First two arguments are treated like torch.Tensor noise and images samples.
        Second two arguments are optional and can be used for latent diffusion models.
        They are treated as latents tensors and conditions.
        Optional 5th element: dict of per-sample GT GS tensors (batch-stacked).

        Args:
            batch: Sequence of tuples ``(noise, images, latents, condition, gt_solver_params?)``.
                Older caches may omit the 5th element.

        Returns:
            Tuple of batched tensors / condition / optional GT dict.
        """
        rows = list(zip(*batch))
        if len(rows) == 5:
            noise, images, latents, condition, gt_list = rows
        elif len(rows) == 4:
            noise, images, latents, condition = rows
            gt_list = None
        else:
            raise ValueError(f"Unexpected batch tuple length {len(rows)}")

        noise = torch.stack(noise)
        images = torch.stack(images)
        latents = torch.stack(latents) if self.config.use_latents else None

        if self.config.use_condition:
            condition = (
                torch.stack(condition)
                if isinstance(condition[0], torch.Tensor)
                else list(condition)
            )
        else:
            condition = None

        gt_batched = None
        if gt_list is not None and gt_list[0] is not None:
            keys = gt_list[0].keys()
            gt_batched = {
                k: torch.stack([sample_gt[k] for sample_gt in gt_list]) for k in keys
            }

        return noise, images, latents, condition, gt_batched
