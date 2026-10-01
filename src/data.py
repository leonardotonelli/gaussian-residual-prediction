"""Preprint implementation: selected components from the research codebase."""
from typing import Dict, Optional
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from .data_source import IWMDataSource
from .mpi3d_data import MPI3DConditionOutcomeSampler, MPI3DDataSource, MPI3DProtocolEpochSampler
from .seed_streams import SeedContext



_DATA_SOURCES: Dict[str, IWMDataSource] = {"mpi3d": MPI3DDataSource()}


def _get_data_source(cfg: Dict[str, object]) -> IWMDataSource:
    """Select a dataset family once; callers then use its shared contract."""
    dataset_name = str(cfg["data"]["dataset"]).lower()
    try:
        return _DATA_SOURCES[dataset_name]
    except KeyError as error:
        supported = ", ".join(sorted(_DATA_SOURCES))
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: {supported}") from error


def maybe_subset_dataset(dataset: Dataset, size: Optional[int]) -> Dataset:
    """Optionally keep only the first size examples for quick experiments."""
    if size is None:
        return dataset
    return Subset(dataset, range(min(size, len(dataset))))


def build_iwm_dataset(
    cfg: Dict[str, object],
    split: str,
    return_label: bool = False,
    return_original: bool = False,
) -> Dataset:
    """Build transition samples through the configured dataset-family adapter."""
    data_cfg = cfg["data"]
    dataset = _get_data_source(cfg).build_transition_dataset(
        cfg,
        split,
        return_label=return_label,
        return_original=return_original,
    )
    if split == "train":
        return maybe_subset_dataset(dataset, data_cfg.get("train_size"))
    return maybe_subset_dataset(dataset, cfg.get("eval", {}).get("size"))


def build_mpi3d_training_sampler(
    cfg: Dict[str, object],
    dataset: Dataset,
    *,
    num_replicas: int = 1,
    rank: int = 0,
) -> Sampler:
    """Build the locked shared source stream and condition-specific outcome stream.

    MPI3D N/S datasets accept ``MPI3DTransitionVisit`` keys rather than plain
    integer indices.  Centralizing this choice prevents a future training
    entrypoint from accidentally using ``shuffle=True`` and introducing an
    uncontrolled outcome path.
    """
    data_cfg = cfg.get("data")
    train_cfg = cfg.get("train")
    if not isinstance(data_cfg, dict) or data_cfg.get("dataset") != "mpi3d":
        raise ValueError("MPI3D training sampler requires data.dataset: mpi3d")
    if not isinstance(train_cfg, dict):
        raise ValueError("MPI3D training sampler requires a train configuration")
    condition = data_cfg.get("condition")
    if condition not in {"D", "N", "S"}:
        raise ValueError("MPI3D training sampler requires data.condition: D, N, or S")
    root_seed = cfg.get("seed")
    if not isinstance(root_seed, int):
        raise ValueError("MPI3D training sampler requires an integer root seed")
    samples_per_epoch = train_cfg.get("transition_samples_per_epoch")
    if not isinstance(samples_per_epoch, int):
        raise ValueError("MPI3D training sampler requires transition_samples_per_epoch")
    sampler_kwargs = {
        "population_size": len(dataset),
        "root_seed": root_seed,
        "samples_per_epoch": samples_per_epoch,
        "num_replicas": num_replicas,
        "rank": rank,
    }
    if "seed_streams" in cfg:
        seed_context = SeedContext.from_dict(cfg["seed_streams"])
        if seed_context.replication != root_seed:
            raise ValueError("MPI3D seed context must match the configured replication")
        sampler_kwargs["seed_context"] = seed_context
    if condition == "D":
        return MPI3DProtocolEpochSampler(**sampler_kwargs)
    outcome_kwargs = {}
    if "execution_success_probability" in data_cfg and condition != "S":
        raise ValueError("data.execution_success_probability is defined only for condition S")
    if "execution_success_probability" in data_cfg:
        outcome_kwargs["success_probability"] = data_cfg["execution_success_probability"]
    if "execution_success_probability_by_action" in data_cfg:
        if condition != "S":
            raise ValueError(
                "data.execution_success_probability_by_action is defined only for condition S"
            )
        outcome_kwargs["success_probability_by_action"] = data_cfg[
            "execution_success_probability_by_action"
        ]
    for factor_label, builder_name in (("size", "source_object_size_ids"), ("shape", "source_object_shape_ids")):
        config_key = f"execution_success_probability_by_action_and_{factor_label}"
        if config_key not in data_cfg:
            continue
        if condition != "S":
            raise ValueError(f"data.{config_key} is defined only for condition S")
        id_builder = getattr(dataset, builder_name, None)
        if not callable(id_builder):
            raise TypeError(
                f"{factor_label}-action outcome sampling requires an MPI3D transition population"
            )
        outcome_kwargs[f"success_probability_by_action_and_{factor_label}"] = data_cfg[
            config_key
        ]
        outcome_kwargs[f"source_{factor_label}_ids"] = id_builder()
    return MPI3DConditionOutcomeSampler(
        condition=condition,
        **outcome_kwargs,
        **sampler_kwargs,
    )


def build_clean_labeled_dataset(
    cfg: Dict[str, object],
    split: str,
    size: Optional[int] = None,
) -> Dataset:
    """Build clean labeled images without imposing a loader ordering policy.

    Frozen-probe protocols may need a fixed extraction order, whereas older
    generic probes shuffle their training images.  Exposing the dataset keeps
    that distinction explicit instead of hiding it in a single loader helper.
    """
    return maybe_subset_dataset(
        _get_data_source(cfg).build_clean_labeled_dataset(cfg, split=split), size
    )


def build_mpi3d_repeated_future_dataloader(cfg: Dict[str, object], split: str) -> DataLoader:
    """Build a fixed N/S target-support loader for prediction evaluation."""
    data_cfg = cfg["data"]
    if data_cfg.get("dataset") != "mpi3d":
        raise ValueError("Repeated-future evaluation is defined only for MPI3D")
    dataset = _get_data_source(cfg).build_repeated_future_dataset(cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
