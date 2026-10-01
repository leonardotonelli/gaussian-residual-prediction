"""Preprint implementation: selected components from the research codebase."""
import torch
from torch.utils.data import Dataset
from .mpi3d_data import MPI3DTaskState, mpi3d_attribute_configurations


def image_partitions(seed=1730, counts=(4096, 1024, 1024), excluded=()):
    """Unique image IDs, not transition IDs: no action-induced image leakage."""
    total = len(mpi3d_attribute_configurations("train")) * 3 * 40 * 40
    available = torch.ones(total, dtype=torch.bool)
    if len(excluded):
        available[torch.tensor(sorted(excluded), dtype=torch.long)] = False
    if len(counts) != 3 or min(counts) <= 0 or sum(counts) > int(available.sum()):
        raise ValueError("Invalid fit/development/holdout image counts")
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed))
    ids = order[available[order]][:sum(counts)]
    return dict(zip(("fit", "development", "holdout"), ids.split(counts)))


class InspectionImages(Dataset):
    """Train attribute combinations, all rendered positions including 0..39."""

    def __init__(self, archive, ids):
        self.archive, self.ids = archive, ids
        self.attributes = mpi3d_attribute_configurations("train")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = int(self.ids[index])
        configuration, remainder = divmod(image_id, 3 * 1600)
        camera, remainder = divmod(remainder, 1600)
        horizontal, vertical = divmod(remainder, 40)
        state = MPI3DTaskState(*self.attributes[configuration], horizontal, vertical)
        return {"image": self.archive.image(state, camera_height=camera),
                "position": torch.tensor([horizontal, vertical], dtype=torch.float32),
                "image_id": image_id}
