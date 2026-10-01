"""MPI3D-realistic factor definitions and deterministic image addressing.

This module intentionally starts below the dataset-loading layer.  Its
functions describe the official MPI3D array layout, the Phase 1 task state,
and the one unambiguous row address for a rendered image.  Keeping this logic
pure makes the scientific assumptions testable before a multi-gigabyte archive
is accessed.
"""

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
from math import prod
from pathlib import Path
import random
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .seed_streams import SeedContext


MPI3D_FACTOR_NAMES = (
    "object_color",
    "object_shape",
    "object_size",
    "camera_height",
    "background_color",
    "horizontal_axis",
    "vertical_axis",
)
MPI3D_FACTOR_SIZES = (6, 6, 2, 3, 3, 40, 40)
MPI3D_IMAGE_SIZE = 64
MPI3D_IMAGE_CHANNELS = 3
MPI3D_BACKGROUND_ID = 0
MPI3D_NUM_IMAGES = prod(MPI3D_FACTOR_SIZES)
MPI3D_IMAGES_ARRAY_SHAPE = (
    MPI3D_NUM_IMAGES,
    MPI3D_IMAGE_SIZE,
    MPI3D_IMAGE_SIZE,
    MPI3D_IMAGE_CHANNELS,
)
MPI3D_ACTION_STRIDE = 4
MPI3D_COMMON_SOURCE_MIN = MPI3D_ACTION_STRIDE
MPI3D_COMMON_SOURCE_MAX = MPI3D_FACTOR_SIZES[5] - 1 - MPI3D_ACTION_STRIDE
MPI3D_CARDINAL_ACTION_NAMES = ("left", "right", "up", "down")
MPI3D_OBJECT_SIZE_NAMES = ("small", "large")
# MPI3D ``object_shape`` factor indices.  The archive does not record
# semantic shape names, so the levels are referenced by index.
MPI3D_OBJECT_SHAPE_NAMES = tuple(
    f"shape{index}" for index in range(MPI3D_FACTOR_SIZES[1])
)
MPI3D_CARDINAL_ACTION_VECTORS = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}
MPI3D_CORE_CONDITIONS = ("D", "N", "S")
# Outcome streams are intentionally separate from the source-action sampler
# and from one another.  The numerical offsets are public, fixed namespaces;
# they are not model or data hyperparameters.
MPI3D_OUTCOME_STREAM_OFFSETS = {"N": 1_000_003, "S": 2_000_003}
MPI3D_ATTRIBUTE_SPLIT_RESIDUES: Dict[str, Tuple[int, ...]] = {
    "train": (0, 1, 2, 3),
    "validation": (4,),
    "test": (5,),
}
MPI3D_MANIFEST_ROOT_SEED = 20_260_807
MPI3D_POSITION_MANIFEST_SPECS = {
    "probe_train": 256,
    "validation": 128,
    "test": 256,
}
MPI3D_POSITION_MANIFEST_STREAM_SEEDS = {
    "probe_train": MPI3D_MANIFEST_ROOT_SEED,
    "validation": MPI3D_MANIFEST_ROOT_SEED + 1,
    "test": MPI3D_MANIFEST_ROOT_SEED + 2,
}
MPI3D_PROBE_PAIR_HOLDOUT_KIND = "disjoint_spatial_position_pairs"
MPI3D_PROBE_PAIR_HOLDOUT_ROOT_SEED = MPI3D_MANIFEST_ROOT_SEED + 3
MPI3D_TRANSITION_SAMPLES_PER_EPOCH = 65_536


@dataclass(frozen=True)
class MPI3DTaskState:
    """Persistent Phase 1 state, excluding camera and background rendering factors."""

    object_color: int
    object_shape: int
    object_size: int
    horizontal_axis: int
    vertical_axis: int

    def __post_init__(self) -> None:
        _validate_factor_id("object_color", self.object_color, MPI3D_FACTOR_SIZES[0])
        _validate_factor_id("object_shape", self.object_shape, MPI3D_FACTOR_SIZES[1])
        _validate_factor_id("object_size", self.object_size, MPI3D_FACTOR_SIZES[2])
        _validate_factor_id("horizontal_axis", self.horizontal_axis, MPI3D_FACTOR_SIZES[5])
        _validate_factor_id("vertical_axis", self.vertical_axis, MPI3D_FACTOR_SIZES[6])


@dataclass(frozen=True)
class MPI3DTransitionOutcome:
    """One realized rendered future under a locked Phase-1 condition."""

    target_state: MPI3DTaskState
    target_camera_height: int
    execution_succeeded: Optional[bool]

    def __post_init__(self) -> None:
        _validate_factor_id(
            "target_camera_height", self.target_camera_height, MPI3D_FACTOR_SIZES[3]
        )


@dataclass(frozen=True)
class MPI3DTransitionVisit:
    """One source/action population index paired with one training outcome ID.

    This object is designed to be yielded by the later condition-aware sampler
    and accepted by the later condition-aware transition dataset.  Keeping the
    outcome alongside its source index prevents an uncontrolled random draw in
    ``Dataset.__getitem__``.
    """

    source_index: int
    outcome_index: int


class MPI3DArchive:
    """Read indexed MPI3D images through a read-only NumPy memory map.

    The official ``.npz`` archive wraps a single large ``images.npy`` member.
    The extracted ``.npy`` file can be memory-mapped, avoiding a full-array RAM
    allocation while allowing random state-addressed image reads.
    """

    def __init__(self, images_path: Union[Path, str]) -> None:
        self.images_path = Path(images_path)
        if not self.images_path.is_file():
            raise FileNotFoundError(f"MPI3D images array does not exist: {self.images_path}")
        self._images = np.load(self.images_path, mmap_mode="r", allow_pickle=False)
        _validate_mpi3d_images_array(self._images)

    def image(self, state: MPI3DTaskState, *, camera_height: int) -> Tensor:
        """Return one locked Phase 1 rendering as RGB float32 in ``[0, 1]``."""
        row_index = mpi3d_image_row_index(state, camera_height=camera_height)
        image = np.array(self._images[row_index], copy=True)
        return torch.from_numpy(image).permute(2, 0, 1).to(torch.float32).div_(255.0)


def mpi3d_flat_row_index(factor_ids: Sequence[int]) -> int:
    """Return the row address in MPI3D's official row-major image array.

    ``factor_ids`` must follow ``MPI3D_FACTOR_NAMES`` exactly.  This is
    equivalent to indexing an array reshaped to
    ``[6, 6, 2, 3, 3, 40, 40, 64, 64, 3]`` and then flattening its first seven
    axes in C order.
    """
    if len(factor_ids) != len(MPI3D_FACTOR_SIZES):
        raise ValueError(
            "MPI3D factor_ids must contain exactly seven IDs in official factor order"
        )

    row_index = 0
    for factor_name, factor_id, factor_size in zip(
        MPI3D_FACTOR_NAMES,
        factor_ids,
        MPI3D_FACTOR_SIZES,
    ):
        _validate_factor_id(factor_name, factor_id, factor_size)
        row_index = row_index * factor_size + factor_id
    return row_index


def mpi3d_render_factor_ids(
    state: MPI3DTaskState,
    *,
    camera_height: int,
) -> Tuple[int, int, int, int, int, int, int]:
    """Build the one permitted Phase 1 rendering address for a task state.

    The function fixes background color to the locked ID zero.  Camera height
    is supplied because it controls appearance but is not part of the task
    state or transition outcome.
    """
    _validate_factor_id("camera_height", camera_height, MPI3D_FACTOR_SIZES[3])
    return (
        state.object_color,
        state.object_shape,
        state.object_size,
        camera_height,
        MPI3D_BACKGROUND_ID,
        state.horizontal_axis,
        state.vertical_axis,
    )


def mpi3d_image_row_index(state: MPI3DTaskState, *, camera_height: int) -> int:
    """Return the flat MPI3D row for a locked Phase 1 image rendering."""
    return mpi3d_flat_row_index(mpi3d_render_factor_ids(state, camera_height=camera_height))


def is_mpi3d_common_legal_source_position(state: MPI3DTaskState) -> bool:
    """Return whether all four stride-four cardinal commands are legal from ``state``."""
    return (
        MPI3D_COMMON_SOURCE_MIN <= state.horizontal_axis <= MPI3D_COMMON_SOURCE_MAX
        and MPI3D_COMMON_SOURCE_MIN <= state.vertical_axis <= MPI3D_COMMON_SOURCE_MAX
    )


def deterministic_mpi3d_target_state(
    source_state: MPI3DTaskState,
    *,
    action_name: str,
) -> MPI3DTaskState:
    """Apply one locked D-condition cardinal translation to a legal source state.

    The action has a unit network representation but moves the physical MPI3D
    position by four factor indices.  Object attributes persist unchanged.
    ``source_state`` must lie in the common legal region so every command has
    one valid target and no action-dependent boundary handling is needed.
    """
    if action_name not in MPI3D_CARDINAL_ACTION_VECTORS:
        supported = ", ".join(MPI3D_CARDINAL_ACTION_NAMES)
        raise ValueError(f"Unknown MPI3D cardinal action {action_name!r}; supported: {supported}")
    if not is_mpi3d_common_legal_source_position(source_state):
        raise ValueError(
            "MPI3D source positions must lie in the common legal region "
            f"[{MPI3D_COMMON_SOURCE_MIN}, {MPI3D_COMMON_SOURCE_MAX}] for both axes"
        )

    delta_horizontal, delta_vertical = MPI3D_CARDINAL_ACTION_VECTORS[action_name]
    return replace(
        source_state,
        horizontal_axis=source_state.horizontal_axis + delta_horizontal * MPI3D_ACTION_STRIDE,
        vertical_axis=source_state.vertical_axis + delta_vertical * MPI3D_ACTION_STRIDE,
    )


def mpi3d_transition_outcome(
    source_state: MPI3DTaskState,
    *,
    source_camera_height: int,
    action_name: str,
    condition: str,
    outcome_index: int,
) -> MPI3DTransitionOutcome:
    """Resolve one explicit training outcome ID into its rendered future.

    Outcome IDs are part of the locked data contract:

    - D: ``0`` is successful execution at the source camera height;
    - N: ``0``, ``1``, or ``2`` is successful execution at that target camera;
    - S: ``0`` is execution failure and ``1`` is successful execution, both at
      the source camera height.

    The outcome is supplied by a deterministic epoch sampler rather than
    sampled here.  Thus image retrieval has no hidden stochasticity.
    """
    _validate_factor_id("source_camera_height", source_camera_height, MPI3D_FACTOR_SIZES[3])
    if condition not in MPI3D_CORE_CONDITIONS:
        supported = ", ".join(MPI3D_CORE_CONDITIONS)
        raise ValueError(f"Unknown MPI3D core condition {condition!r}; supported: {supported}")
    successful_target = deterministic_mpi3d_target_state(source_state, action_name=action_name)
    if condition == "D":
        if outcome_index != 0:
            raise ValueError("MPI3D D requires outcome_index: 0")
        return MPI3DTransitionOutcome(
            target_state=successful_target,
            target_camera_height=source_camera_height,
            execution_succeeded=True,
        )
    if condition == "N":
        _validate_factor_id("N target_camera_height outcome_index", outcome_index, MPI3D_FACTOR_SIZES[3])
        return MPI3DTransitionOutcome(
            target_state=successful_target,
            target_camera_height=outcome_index,
            execution_succeeded=True,
        )

    if outcome_index not in {0, 1}:
        raise ValueError("MPI3D S requires outcome_index 0 (failure) or 1 (success)")
    return MPI3DTransitionOutcome(
        target_state=successful_target if outcome_index == 1 else source_state,
        target_camera_height=source_camera_height,
        execution_succeeded=bool(outcome_index),
    )


def mpi3d_valid_transition_outcomes(
    source_state: MPI3DTaskState,
    *,
    source_camera_height: int,
    action_name: str,
    condition: str,
) -> Tuple[MPI3DTransitionOutcome, ...]:
    """Enumerate every valid rendered future for N/S validation and test.

    This is not a training target sampler.  It makes all valid futures
    observable to the repeated-future evaluation layer, so later metrics never
    score a legitimate N camera or S failure as an erroneous target.
    """
    if condition not in MPI3D_CORE_CONDITIONS:
        supported = ", ".join(MPI3D_CORE_CONDITIONS)
        raise ValueError(f"Unknown MPI3D core condition {condition!r}; supported: {supported}")
    num_outcomes = {"D": 1, "N": MPI3D_FACTOR_SIZES[3], "S": 2}[condition]
    return tuple(
        mpi3d_transition_outcome(
            source_state,
            source_camera_height=source_camera_height,
            action_name=action_name,
            condition=condition,
            outcome_index=outcome_index,
        )
        for outcome_index in range(num_outcomes)
    )


def mpi3d_outcome_stream_seed(
    *, root_seed: int, condition: str, epoch_index: int, seed_context: Optional[SeedContext] = None
) -> int:
    """Return the public deterministic outcome-stream seed for one epoch."""
    if not 0 <= root_seed < 2**63:
        raise ValueError("MPI3D outcome root_seed must be in [0, 2**63)")
    if not isinstance(epoch_index, int) or epoch_index < 0:
        raise ValueError("MPI3D outcome epoch_index must be a non-negative integer")
    if condition not in MPI3D_OUTCOME_STREAM_OFFSETS:
        supported = ", ".join(MPI3D_OUTCOME_STREAM_OFFSETS)
        raise ValueError(f"MPI3D outcome streams exist only for N/S, not {condition!r}; supported: {supported}")
    if seed_context is not None:
        _validate_mpi3d_training_context(seed_context)
        return seed_context.seed("data-outcome", partition=condition, epoch=epoch_index)
    outcome_seed = root_seed + MPI3D_OUTCOME_STREAM_OFFSETS[condition] + epoch_index
    if outcome_seed >= 2**63:
        raise ValueError("MPI3D outcome stream seed exceeds torch's supported range")
    return outcome_seed


def _validate_mpi3d_training_context(seed_context: SeedContext) -> None:
    if seed_context.dataset != "mpi3d" or seed_context.purpose == "evaluation":
        raise ValueError("MPI3D training requires an MPI3D training SeedContext")


def mpi3d_s_success_count_for_epoch(
    *, samples_per_epoch: int, success_probability: float, epoch_index: int
) -> int:
    """Allocate a deterministic binary-outcome quota with exact cumulative mass.

    A probability such as ``0.2`` cannot be represented exactly inside one
    65,536-example epoch.  The cumulative-floor construction makes the count
    error smaller than one example at every prefix and exact whenever the
    cumulative sample count is divisible by the probability denominator.
    """
    if not isinstance(samples_per_epoch, int) or samples_per_epoch <= 0:
        raise ValueError("samples_per_epoch must be a positive integer")
    if not isinstance(epoch_index, int) or epoch_index < 0:
        raise ValueError("epoch_index must be a non-negative integer")
    if isinstance(success_probability, bool) or not isinstance(
        success_probability, (int, float)
    ):
        raise ValueError("success_probability must be a number strictly between zero and one")
    probability = Fraction(str(success_probability))
    if probability <= 0 or probability >= 1:
        raise ValueError("success_probability must be strictly between zero and one")
    numerator = probability.numerator
    denominator = probability.denominator
    cumulative_before = epoch_index * samples_per_epoch * numerator // denominator
    cumulative_after = (epoch_index + 1) * samples_per_epoch * numerator // denominator
    return cumulative_after - cumulative_before


def _normalize_state_action_probabilities(
    table: Optional[Mapping[str, Mapping[str, float]]],
    *,
    source_ids: Optional[Tensor],
    level_names: Tuple[str, ...],
    factor_label: str,
    condition: str,
    success_probability: float,
    population_size: int,
) -> Tuple[Optional[Dict[str, Dict[str, float]]], Optional[Tensor]]:
    """Validate one state-factor x action success-probability table.

    ``factor_label`` names the conditioning factor ("size" or "shape") purely
    for error messages.  Returning ``(None, None)`` keeps an unconfigured
    factor inert.
    """
    if table is None:
        if source_ids is not None:
            raise ValueError(
                f"source_{factor_label}_ids require {factor_label}-action probabilities"
            )
        return None, None
    if condition != "S":
        raise ValueError(
            f"{factor_label}-action success probabilities require condition S"
        )
    if success_probability != 0.5:
        raise ValueError(
            "configure either global success_probability or "
            f"{factor_label}-action probabilities"
        )
    if set(table) != set(level_names):
        expected = "/".join(level_names)
        raise ValueError(
            f"{factor_label}-action probabilities must define exactly {expected}"
        )
    normalized: Dict[str, Dict[str, float]] = {}
    for level_name in level_names:
        action_probabilities = table[level_name]
        if set(action_probabilities) != set(MPI3D_CARDINAL_ACTION_NAMES):
            raise ValueError(
                f"each {factor_label}-action table must define exactly left/right/up/down"
            )
        normalized[level_name] = {
            action_name: float(action_probabilities[action_name])
            for action_name in MPI3D_CARDINAL_ACTION_NAMES
        }
    if any(
        not 0.0 < probability < 1.0
        for probabilities in normalized.values()
        for probability in probabilities.values()
    ):
        raise ValueError(
            f"every {factor_label}-action success probability must be in (0, 1)"
        )
    if source_ids is None:
        raise ValueError(
            f"{factor_label}-action probabilities require source_{factor_label}_ids"
        )
    normalized_ids = torch.as_tensor(source_ids, dtype=torch.long).clone()
    if normalized_ids.shape != (population_size,):
        raise ValueError(
            f"source_{factor_label}_ids must have shape [population_size]"
        )
    if not ((normalized_ids >= 0) & (normalized_ids < len(level_names))).all():
        raise ValueError(
            f"source_{factor_label}_ids must contain only IDs 0..{len(level_names) - 1}"
        )
    return normalized, normalized_ids


def mpi3d_attribute_residue(state: MPI3DTaskState) -> int:
    """Return the locked six-way object-attribute split residue for ``state``."""
    return (state.object_color + state.object_shape + 3 * state.object_size) % 6


def mpi3d_attribute_split(state: MPI3DTaskState) -> str:
    """Assign a persistent task state to its locked train/validation/test split."""
    residue = mpi3d_attribute_residue(state)
    for split, residues in MPI3D_ATTRIBUTE_SPLIT_RESIDUES.items():
        if residue in residues:
            return split
    raise RuntimeError(f"No MPI3D attribute split contains residue {residue}")


def mpi3d_attribute_configurations(split: str) -> Tuple[Tuple[int, int, int], ...]:
    """Enumerate the object-attribute configurations assigned to one split."""
    if split not in MPI3D_ATTRIBUTE_SPLIT_RESIDUES:
        supported = ", ".join(MPI3D_ATTRIBUTE_SPLIT_RESIDUES)
        raise ValueError(f"Unknown MPI3D attribute split {split!r}; supported: {supported}")
    configurations = tuple(
        (object_color, object_shape, object_size)
        for object_color in range(MPI3D_FACTOR_SIZES[0])
        for object_shape in range(MPI3D_FACTOR_SIZES[1])
        for object_size in range(MPI3D_FACTOR_SIZES[2])
        if mpi3d_attribute_split(
            MPI3DTaskState(object_color, object_shape, object_size, 4, 4)
        )
        == split
    )
    return configurations


class MPI3DDeterministicTransitionDataset(Dataset):
    """Complete D-condition population over one attribute split and source positions.

    Each index deterministically identifies one source state, camera height,
    and cardinal command.  This fixed population is intentionally separate
    from the later epoch sampler, which draws its without-replacement subset.
    """

    def __init__(
        self,
        archive: MPI3DArchive,
        *,
        split: str,
        positions: Optional[Iterable[Sequence[int]]] = None,
    ) -> None:
        self.archive = archive
        self.split = split
        self.attribute_configurations = mpi3d_attribute_configurations(split)
        self.positions = (
            _all_mpi3d_common_legal_positions()
            if positions is None
            else _validate_unique_legal_mpi3d_positions(positions)
        )
        self.num_actions = len(MPI3D_CARDINAL_ACTION_NAMES)
        self.num_camera_heights = MPI3D_FACTOR_SIZES[3]

    def __len__(self) -> int:
        return (
            len(self.attribute_configurations)
            * self.num_camera_heights
            * len(self.positions)
            * self.num_actions
        )

    def _source_components_at(self, index: int) -> Tuple[MPI3DTaskState, int, str]:
        """Decode one fixed population index without choosing an outcome."""
        if not 0 <= index < len(self):
            raise IndexError(f"MPI3D transition index {index} is outside [0, {len(self)})")
        action_index = index % self.num_actions
        position_index = (index // self.num_actions) % len(self.positions)
        camera_height = (index // (self.num_actions * len(self.positions))) % self.num_camera_heights
        configuration_index = index // (
            self.num_actions * len(self.positions) * self.num_camera_heights
        )
        object_color, object_shape, object_size = self.attribute_configurations[configuration_index]
        horizontal_axis, vertical_axis = self.positions[position_index]
        source_state = MPI3DTaskState(
            object_color,
            object_shape,
            object_size,
            horizontal_axis,
            vertical_axis,
        )
        action_name = MPI3D_CARDINAL_ACTION_NAMES[action_index]
        return source_state, camera_height, action_name

    def _source_attribute_ids(self, configuration_index: int, label: str) -> Tensor:
        """Return one object-attribute ID for every fixed source/action index."""
        sources_per_configuration = (
            self.num_camera_heights * len(self.positions) * self.num_actions
        )
        configuration_values = torch.tensor(
            [
                configuration[configuration_index]
                for configuration in self.attribute_configurations
            ],
            dtype=torch.long,
        )
        attribute_ids = configuration_values.repeat_interleave(sources_per_configuration)
        if attribute_ids.shape != (len(self),):
            raise RuntimeError(f"MPI3D source-{label} index construction is inconsistent")
        return attribute_ids

    def source_object_size_ids(self) -> Tensor:
        """Return the object-size ID for every fixed source/action index."""
        return self._source_attribute_ids(2, "size")

    def source_object_shape_ids(self) -> Tensor:
        """Return the object-shape ID for every fixed source/action index."""
        return self._source_attribute_ids(1, "shape")

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        source_state, camera_height, action_name = self._source_components_at(index)
        target_state = deterministic_mpi3d_target_state(source_state, action_name=action_name)
        action_vector = MPI3D_CARDINAL_ACTION_VECTORS[action_name]

        return {
            "x_source": self.archive.image(source_state, camera_height=camera_height),
            "y_target": self.archive.image(target_state, camera_height=camera_height),
            "action": torch.tensor(action_vector, dtype=torch.float32),
            "source_factors": torch.tensor(
                mpi3d_render_factor_ids(source_state, camera_height=camera_height),
                dtype=torch.long,
            ),
            "target_factors": torch.tensor(
                mpi3d_render_factor_ids(target_state, camera_height=camera_height),
                dtype=torch.long,
            ),
            "source_position": torch.tensor(
                [source_state.horizontal_axis, source_state.vertical_axis], dtype=torch.long
            ),
            "target_position": torch.tensor(
                [target_state.horizontal_axis, target_state.vertical_axis], dtype=torch.long
            ),
        }


class MPI3DConditionTransitionDataset(MPI3DDeterministicTransitionDataset):
    """N/S training pairs addressed by explicit source/outcome visits.

    The superclass defines the shared, fixed source/action population.  A
    caller must pass a :class:`MPI3DTransitionVisit` produced by the
    condition-aware sampler, so this dataset never makes an implicit random
    environment draw in ``__getitem__``.
    """

    def __init__(
        self,
        archive: MPI3DArchive,
        *,
        split: str,
        condition: str,
        positions: Optional[Iterable[Sequence[int]]] = None,
    ) -> None:
        if condition not in {"N", "S"}:
            raise ValueError("MPI3D condition transition pairs require condition 'N' or 'S'")
        super().__init__(archive, split=split, positions=positions)
        self.condition = condition

    def __getitem__(self, visit: MPI3DTransitionVisit) -> Dict[str, Tensor]:
        if not isinstance(visit, MPI3DTransitionVisit):
            raise TypeError(
                "MPI3D condition transition pairs require an MPI3DTransitionVisit "
                "from MPI3DConditionOutcomeSampler"
            )
        source_state, source_camera_height, action_name = self._source_components_at(
            visit.source_index
        )
        outcome = mpi3d_transition_outcome(
            source_state,
            source_camera_height=source_camera_height,
            action_name=action_name,
            condition=self.condition,
            outcome_index=visit.outcome_index,
        )
        action_vector = MPI3D_CARDINAL_ACTION_VECTORS[action_name]
        return {
            "x_source": self.archive.image(source_state, camera_height=source_camera_height),
            "y_target": self.archive.image(
                outcome.target_state, camera_height=outcome.target_camera_height
            ),
            "action": torch.tensor(action_vector, dtype=torch.float32),
            "source_factors": torch.tensor(
                mpi3d_render_factor_ids(source_state, camera_height=source_camera_height),
                dtype=torch.long,
            ),
            "target_factors": torch.tensor(
                mpi3d_render_factor_ids(
                    outcome.target_state, camera_height=outcome.target_camera_height
                ),
                dtype=torch.long,
            ),
            "source_position": torch.tensor(
                [source_state.horizontal_axis, source_state.vertical_axis], dtype=torch.long
            ),
            "target_position": torch.tensor(
                [outcome.target_state.horizontal_axis, outcome.target_state.vertical_axis],
                dtype=torch.long,
            ),
            "outcome_index": torch.tensor(visit.outcome_index, dtype=torch.long),
            "execution_succeeded": torch.tensor(outcome.execution_succeeded, dtype=torch.bool),
        }


class MPI3DRepeatedFutureDataset(MPI3DDeterministicTransitionDataset):
    """Fixed N/S validation or test queries with every valid target future.

    Each item represents one source image and action.  The target-view axis is
    the complete future support in the public outcome-ID order: camera heights
    ``0, 1, 2`` for N and failure/success for S.  This is intentionally not a
    training dataset and has no stochastic sampling path.
    """

    def __init__(
        self,
        archive: MPI3DArchive,
        *,
        split: str,
        condition: str,
        positions: Iterable[Sequence[int]],
    ) -> None:
        if condition not in {"N", "S"}:
            raise ValueError("MPI3D repeated futures require condition 'N' or 'S'")
        if split not in {"validation", "test"}:
            raise ValueError("MPI3D repeated futures are defined only for validation and test")
        super().__init__(archive, split=split, positions=positions)
        self.condition = condition

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        source_state, source_camera_height, action_name = self._source_components_at(index)
        outcomes = mpi3d_valid_transition_outcomes(
            source_state,
            source_camera_height=source_camera_height,
            action_name=action_name,
            condition=self.condition,
        )
        return {
            "x_source": self.archive.image(source_state, camera_height=source_camera_height),
            "action": torch.tensor(
                MPI3D_CARDINAL_ACTION_VECTORS[action_name], dtype=torch.float32
            ),
            "source_factors": torch.tensor(
                mpi3d_render_factor_ids(source_state, camera_height=source_camera_height),
                dtype=torch.long,
            ),
            "source_position": torch.tensor(
                [source_state.horizontal_axis, source_state.vertical_axis], dtype=torch.long
            ),
            "target_views": torch.stack(
                [
                    self.archive.image(
                        outcome.target_state, camera_height=outcome.target_camera_height
                    )
                    for outcome in outcomes
                ]
            ),
            "candidate_target_factors": torch.tensor(
                [
                    mpi3d_render_factor_ids(
                        outcome.target_state, camera_height=outcome.target_camera_height
                    )
                    for outcome in outcomes
                ],
                dtype=torch.long,
            ),
            "candidate_target_positions": torch.tensor(
                [
                    [outcome.target_state.horizontal_axis, outcome.target_state.vertical_axis]
                    for outcome in outcomes
                ],
                dtype=torch.long,
            ),
            "candidate_outcome_indices": torch.arange(len(outcomes), dtype=torch.long),
            "candidate_execution_succeeded": torch.tensor(
                [outcome.execution_succeeded for outcome in outcomes], dtype=torch.bool
            ),
        }


class MPI3DDeterministicKViewDataset(Dataset):
    """Four-action target banks for D-condition action retrieval evaluation."""

    def __init__(
        self,
        archive: MPI3DArchive,
        *,
        split: str,
        positions: Iterable[Sequence[int]],
    ) -> None:
        self.archive = archive
        self.split = split
        self.attribute_configurations = mpi3d_attribute_configurations(split)
        self.positions = _validate_unique_legal_mpi3d_positions(positions)
        self.num_actions = len(MPI3D_CARDINAL_ACTION_NAMES)
        self.num_camera_heights = MPI3D_FACTOR_SIZES[3]

    def __len__(self) -> int:
        return (
            len(self.attribute_configurations)
            * self.num_camera_heights
            * len(self.positions)
            * self.num_actions
        )

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(f"MPI3D K-view index {index} is outside [0, {len(self)})")
        action_index = index % self.num_actions
        position_index = (index // self.num_actions) % len(self.positions)
        camera_height = (index // (self.num_actions * len(self.positions))) % self.num_camera_heights
        configuration_index = index // (
            self.num_actions * len(self.positions) * self.num_camera_heights
        )
        object_color, object_shape, object_size = self.attribute_configurations[configuration_index]
        horizontal_axis, vertical_axis = self.positions[position_index]
        source_state = MPI3DTaskState(
            object_color,
            object_shape,
            object_size,
            horizontal_axis,
            vertical_axis,
        )
        query_action_name = MPI3D_CARDINAL_ACTION_NAMES[action_index]
        candidate_action_names = (query_action_name,) + tuple(
            action_name
            for action_name in MPI3D_CARDINAL_ACTION_NAMES
            if action_name != query_action_name
        )
        candidate_target_states = tuple(
            deterministic_mpi3d_target_state(source_state, action_name=action_name)
            for action_name in candidate_action_names
        )

        return {
            "x_source": self.archive.image(source_state, camera_height=camera_height),
            "action": torch.tensor(
                MPI3D_CARDINAL_ACTION_VECTORS[query_action_name], dtype=torch.float32
            ),
            "action_name": query_action_name,
            "target_views": torch.stack(
                [
                    self.archive.image(target_state, camera_height=camera_height)
                    for target_state in candidate_target_states
                ]
            ),
            "source_factors": torch.tensor(
                mpi3d_render_factor_ids(source_state, camera_height=camera_height),
                dtype=torch.long,
            ),
            "candidate_target_factors": torch.tensor(
                [
                    mpi3d_render_factor_ids(target_state, camera_height=camera_height)
                    for target_state in candidate_target_states
                ],
                dtype=torch.long,
            ),
        }


class MPI3DCleanImageDataset(Dataset):
    """Clean MPI3D images and complete factor metadata for frozen probing."""

    def __init__(
        self,
        archive: MPI3DArchive,
        *,
        split: str,
        positions: Iterable[Sequence[int]],
    ) -> None:
        self.archive = archive
        self.split = split
        self.attribute_configurations = mpi3d_attribute_configurations(split)
        self.positions = _validate_unique_legal_mpi3d_positions(positions)
        self.num_camera_heights = MPI3D_FACTOR_SIZES[3]

    def __len__(self) -> int:
        return len(self.attribute_configurations) * self.num_camera_heights * len(self.positions)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(f"MPI3D clean-image index {index} is outside [0, {len(self)})")
        position_index = index % len(self.positions)
        camera_height = (index // len(self.positions)) % self.num_camera_heights
        configuration_index = index // (len(self.positions) * self.num_camera_heights)
        object_color, object_shape, object_size = self.attribute_configurations[configuration_index]
        horizontal_axis, vertical_axis = self.positions[position_index]
        state = MPI3DTaskState(
            object_color,
            object_shape,
            object_size,
            horizontal_axis,
            vertical_axis,
        )

        return {
            "image": self.archive.image(state, camera_height=camera_height),
            "factors": torch.tensor(
                mpi3d_render_factor_ids(state, camera_height=camera_height),
                dtype=torch.long,
            ),
            "position": torch.tensor([horizontal_axis, vertical_axis], dtype=torch.long),
            "camera_height": torch.tensor(camera_height, dtype=torch.long),
            "shape_id": torch.tensor(object_shape, dtype=torch.long),
            "color_id": torch.tensor(object_color, dtype=torch.long),
            "size_id": torch.tensor(object_size, dtype=torch.long),
        }


class MPI3DProtocolEpochSampler(Sampler[int]):
    """Fresh without-replacement MPI3D tuple draws for one protocol epoch.

    The sampler is independent of a transition condition.  Matching numerical
    model seeds therefore produce the same ordered source-action indices for D,
    N, and S, while every epoch uses a separate deterministic random stream.
    """

    def __init__(
        self,
        *,
        population_size: int,
        root_seed: int,
        samples_per_epoch: int = MPI3D_TRANSITION_SAMPLES_PER_EPOCH,
        num_replicas: int = 1,
        rank: int = 0,
        seed_context: Optional[SeedContext] = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError("MPI3D sampler population_size must be positive")
        if not 0 <= root_seed < 2**63:
            raise ValueError("MPI3D sampler root_seed must be in [0, 2**63)")
        if not 0 < samples_per_epoch <= population_size:
            raise ValueError("MPI3D samples_per_epoch must be in [1, population_size]")
        if not isinstance(num_replicas, int) or num_replicas <= 0:
            raise ValueError("MPI3D sampler num_replicas must be a positive integer")
        if not isinstance(rank, int) or not 0 <= rank < num_replicas:
            raise ValueError("MPI3D sampler rank must be in [0, num_replicas)")
        if samples_per_epoch % num_replicas != 0:
            raise ValueError(
                "MPI3D samples_per_epoch must divide evenly across distributed sampler replicas"
            )
        if seed_context is not None:
            _validate_mpi3d_training_context(seed_context)
        self.seed_context = seed_context
        self.population_size = population_size
        self.root_seed = root_seed
        self.samples_per_epoch = samples_per_epoch
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch_index = 0

    def set_epoch(self, epoch_index: int) -> None:
        if not isinstance(epoch_index, int) or epoch_index < 0:
            raise ValueError("MPI3D sampler epoch_index must be a non-negative integer")
        self.epoch_index = epoch_index

    def indices_for_epoch(self, epoch_index: int) -> Tensor:
        """Return the ordered, unique population indices for ``epoch_index``."""
        epoch_seed = self.epoch_seed(epoch_index)
        generator = torch.Generator().manual_seed(epoch_seed)
        return torch.randperm(self.population_size, generator=generator)[: self.samples_per_epoch]

    def epoch_seed(self, epoch_index: int) -> int:
        """Return the recorded deterministic stream seed for one protocol epoch."""
        if not isinstance(epoch_index, int) or epoch_index < 0:
            raise ValueError("MPI3D sampler epoch_index must be a non-negative integer")
        if self.seed_context is not None:
            return self.seed_context.seed("data-source-order", partition="train", epoch=epoch_index)
        if self.root_seed + epoch_index >= 2**63:
            raise ValueError("MPI3D sampler epoch seed exceeds torch's supported range")
        return self.root_seed + epoch_index

    def metadata_for_epoch(self, epoch_index: int) -> Dict[str, object]:
        """Record reproducibility metadata without storing all sampled indices twice."""
        indices = self.indices_for_epoch(epoch_index)
        local_indices = indices[self.rank :: self.num_replicas]
        metadata = {
            "epoch_index": epoch_index,
            "epoch_seed": self.epoch_seed(epoch_index),
            "population_size": self.population_size,
            "samples_per_epoch": self.samples_per_epoch,
            "ordered_indices_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "local_samples_per_epoch": len(local_indices),
            "local_ordered_indices_sha256": hashlib.sha256(
                local_indices.numpy().tobytes()
            ).hexdigest(),
        }
        if self.seed_context is not None:
            metadata["seed_streams"] = self.seed_context.as_dict()
            metadata["source_stream_key"] = self.seed_context.key(
                "data-source-order", partition="train", epoch=epoch_index
            ).record()
            metadata["sharding"] = "global_order_then_rank_stride"
        return metadata

    def __iter__(self):
        return iter(
            self.indices_for_epoch(self.epoch_index)[self.rank :: self.num_replicas].tolist()
        )

    def __len__(self) -> int:
        return self.samples_per_epoch // self.num_replicas


class MPI3DConditionOutcomeSampler(Sampler[MPI3DTransitionVisit]):
    """Pair the shared source-action stream with controlled N/S outcomes.

    The source indices are exactly those of :class:`MPI3DProtocolEpochSampler`.
    Only the paired outcome IDs vary by condition.  Outcome assignment happens
    once globally, is shuffled with a condition-specific deterministic stream,
    and is then interleaved across ranks in the same way as source indices.

    Existing S configurations default to exact 50/50 outcomes.  A separately
    documented prevalence-shift follow-up may supply another global success
    probability; cumulative quotas preserve its probability over the run
    without changing the number of training updates.
    """

    def __init__(
        self,
        *,
        population_size: int,
        root_seed: int,
        condition: str,
        success_probability: float = 0.5,
        success_probability_by_action: Optional[Mapping[str, float]] = None,
        success_probability_by_action_and_size: Optional[
            Mapping[str, Mapping[str, float]]
        ] = None,
        source_size_ids: Optional[Tensor] = None,
        success_probability_by_action_and_shape: Optional[
            Mapping[str, Mapping[str, float]]
        ] = None,
        source_shape_ids: Optional[Tensor] = None,
        samples_per_epoch: int = MPI3D_TRANSITION_SAMPLES_PER_EPOCH,
        num_replicas: int = 1,
        rank: int = 0,
        seed_context: Optional[SeedContext] = None,
    ) -> None:
        if condition not in MPI3D_CORE_CONDITIONS:
            supported = ", ".join(MPI3D_CORE_CONDITIONS)
            raise ValueError(f"Unknown MPI3D core condition {condition!r}; supported: {supported}")
        if condition != "S" and (
            success_probability != 0.5
            or success_probability_by_action is not None
            or success_probability_by_action_and_size is not None
            or success_probability_by_action_and_shape is not None
        ):
            raise ValueError("success probabilities are defined only for MPI3D condition S")
        if (
            sum(
                table is not None
                for table in (
                    success_probability_by_action,
                    success_probability_by_action_and_size,
                    success_probability_by_action_and_shape,
                )
            )
            > 1
        ):
            raise ValueError(
                "configure action-only, size-action, or shape-action probabilities, not several"
            )
        if success_probability_by_action is not None:
            if condition != "S":
                raise ValueError("action-dependent success probabilities require condition S")
            if success_probability != 0.5:
                raise ValueError(
                    "configure either global success_probability or "
                    "success_probability_by_action, not both"
                )
            if set(success_probability_by_action) != set(MPI3D_CARDINAL_ACTION_NAMES):
                raise ValueError(
                    "success_probability_by_action must define exactly left/right/up/down"
                )
            normalized_action_probabilities = {
                name: float(success_probability_by_action[name])
                for name in MPI3D_CARDINAL_ACTION_NAMES
            }
            if any(not 0.0 < value < 1.0 for value in normalized_action_probabilities.values()):
                raise ValueError("every action-dependent success probability must be in (0, 1)")
        else:
            normalized_action_probabilities = None
        normalized_size_action_probabilities, normalized_source_size_ids = (
            _normalize_state_action_probabilities(
                success_probability_by_action_and_size,
                source_ids=source_size_ids,
                level_names=MPI3D_OBJECT_SIZE_NAMES,
                factor_label="size",
                condition=condition,
                success_probability=success_probability,
                population_size=population_size,
            )
        )
        normalized_shape_action_probabilities, normalized_source_shape_ids = (
            _normalize_state_action_probabilities(
                success_probability_by_action_and_shape,
                source_ids=source_shape_ids,
                level_names=MPI3D_OBJECT_SHAPE_NAMES,
                factor_label="shape",
                condition=condition,
                success_probability=success_probability,
                population_size=population_size,
            )
        )
        uses_global_probability = (
            normalized_action_probabilities is None
            and normalized_size_action_probabilities is None
            and normalized_shape_action_probabilities is None
        )
        if (
            condition == "S"
            and uses_global_probability
            and success_probability == 0.5
            and samples_per_epoch % 2
        ):
            raise ValueError("MPI3D S samples_per_epoch must be even for exact success/failure balance")
        if condition == "S" and uses_global_probability:
            mpi3d_s_success_count_for_epoch(
                samples_per_epoch=samples_per_epoch,
                success_probability=success_probability,
                epoch_index=0,
            )
        self.source_sampler = MPI3DProtocolEpochSampler(
            population_size=population_size,
            root_seed=root_seed,
            samples_per_epoch=samples_per_epoch,
            num_replicas=num_replicas,
            rank=rank,
            seed_context=seed_context,
        )
        self.seed_context = seed_context
        self.condition = condition
        self.success_probability = (
            float(success_probability)
            if condition == "S" and uses_global_probability
            else None
        )
        self.success_probability_by_action = normalized_action_probabilities
        self.success_probability_by_action_and_size = (
            normalized_size_action_probabilities
        )
        self.source_size_ids = normalized_source_size_ids
        self.success_probability_by_action_and_shape = (
            normalized_shape_action_probabilities
        )
        self.source_shape_ids = normalized_source_shape_ids
        self.root_seed = root_seed
        self.epoch_index = 0

    @property
    def population_size(self) -> int:
        return self.source_sampler.population_size

    @property
    def samples_per_epoch(self) -> int:
        return self.source_sampler.samples_per_epoch

    @property
    def num_replicas(self) -> int:
        return self.source_sampler.num_replicas

    @property
    def rank(self) -> int:
        return self.source_sampler.rank

    def set_epoch(self, epoch_index: int) -> None:
        self.source_sampler.set_epoch(epoch_index)
        self.epoch_index = epoch_index

    def active_state_action_probabilities(
        self,
    ) -> Tuple[
        Optional[Dict[str, Dict[str, float]]], Tuple[str, ...], Optional[Tensor]
    ]:
        """Return the configured state-factor table, its levels, and source IDs.

        At most one state factor may condition the outcome law, so this
        collapses the size and shape tables into the single active one.
        """
        if self.success_probability_by_action_and_size is not None:
            return (
                self.success_probability_by_action_and_size,
                MPI3D_OBJECT_SIZE_NAMES,
                self.source_size_ids,
            )
        if self.success_probability_by_action_and_shape is not None:
            return (
                self.success_probability_by_action_and_shape,
                MPI3D_OBJECT_SHAPE_NAMES,
                self.source_shape_ids,
            )
        return None, (), None

    def outcome_indices_for_epoch(self, epoch_index: int) -> Tensor:
        """Return globally ordered balanced outcome IDs for one protocol epoch."""
        if not isinstance(epoch_index, int) or epoch_index < 0:
            raise ValueError("MPI3D outcome epoch_index must be a non-negative integer")
        if self.condition == "D":
            return torch.zeros(self.samples_per_epoch, dtype=torch.long)
        if self.condition == "S":
            state_table, state_level_names, source_state_ids = (
                self.active_state_action_probabilities()
            )
            if self.success_probability_by_action is not None or state_table is not None:
                source_indices = self.source_sampler.indices_for_epoch(epoch_index)
                outcomes = torch.zeros(self.samples_per_epoch, dtype=torch.long)
                generator = torch.Generator().manual_seed(
                    mpi3d_outcome_stream_seed(
                        root_seed=self.root_seed,
                        condition=self.condition,
                        epoch_index=epoch_index,
                        seed_context=self.seed_context,
                    )
                )
                sampled_state_ids = (
                    source_state_ids[source_indices]
                    if source_state_ids is not None
                    else None
                )
                state_groups = (
                    tuple(enumerate(state_level_names))
                    if sampled_state_ids is not None
                    else ((None, None),)
                )
                for state_id, state_name in state_groups:
                    for action_index, action_name in enumerate(
                        MPI3D_CARDINAL_ACTION_NAMES
                    ):
                        selected = source_indices.remainder(
                            len(MPI3D_CARDINAL_ACTION_NAMES)
                        ) == action_index
                        if state_id is not None:
                            selected &= sampled_state_ids == state_id
                            assert state_table is not None
                            probability_value = state_table[state_name][action_name]
                        else:
                            assert self.success_probability_by_action is not None
                            probability_value = self.success_probability_by_action[
                                action_name
                            ]
                        positions = torch.nonzero(selected, as_tuple=False).flatten()
                        probability = Fraction(str(probability_value))
                        success_count = int(
                            len(positions)
                            * probability.numerator
                            // probability.denominator
                        )
                        group_outcomes = torch.cat(
                            (
                                torch.zeros(
                                    len(positions) - success_count, dtype=torch.long
                                ),
                                torch.ones(success_count, dtype=torch.long),
                            )
                        )
                        group_outcomes = group_outcomes[
                            torch.randperm(len(group_outcomes), generator=generator)
                        ]
                        outcomes[positions] = group_outcomes
                return outcomes
            assert self.success_probability is not None
            success_count = mpi3d_s_success_count_for_epoch(
                samples_per_epoch=self.samples_per_epoch,
                success_probability=self.success_probability,
                epoch_index=epoch_index,
            )
            unshuffled = torch.cat(
                (
                    torch.zeros(self.samples_per_epoch - success_count, dtype=torch.long),
                    torch.ones(success_count, dtype=torch.long),
                )
            )
        else:
            # Cycle through target camera IDs before shuffling.  Counts differ
            # by at most one, exactly as the locked N protocol specifies.
            unshuffled = torch.arange(self.samples_per_epoch, dtype=torch.long) % MPI3D_FACTOR_SIZES[3]
        generator = torch.Generator().manual_seed(
            mpi3d_outcome_stream_seed(
                root_seed=self.root_seed, condition=self.condition, epoch_index=epoch_index,
                seed_context=self.seed_context,
            )
        )
        return unshuffled[torch.randperm(self.samples_per_epoch, generator=generator)]

    def visits_for_epoch(self, epoch_index: int) -> Tuple[MPI3DTransitionVisit, ...]:
        """Return this rank's ordered source-index/outcome-ID training visits."""
        source_indices = self.source_sampler.indices_for_epoch(epoch_index)
        outcome_indices = self.outcome_indices_for_epoch(epoch_index)
        local_source_indices = source_indices[self.rank :: self.num_replicas]
        local_outcome_indices = outcome_indices[self.rank :: self.num_replicas]
        return tuple(
            MPI3DTransitionVisit(source_index=int(source_index), outcome_index=int(outcome_index))
            for source_index, outcome_index in zip(local_source_indices, local_outcome_indices)
        )

    def metadata_for_epoch(self, epoch_index: int) -> Dict[str, object]:
        """Record both the shared source stream and condition-specific outcomes."""
        metadata = self.source_sampler.metadata_for_epoch(epoch_index)
        outcome_indices = self.outcome_indices_for_epoch(epoch_index)
        local_outcome_indices = outcome_indices[self.rank :: self.num_replicas]
        counts = torch.bincount(
            outcome_indices,
            minlength={"D": 1, "N": 3, "S": 2}[self.condition],
        )
        metadata.update(
            {
                "condition": self.condition,
                "outcome_root_seed": self.root_seed if self.condition != "D" else None,
                "outcome_epoch_seed": (
                    mpi3d_outcome_stream_seed(
                        root_seed=self.root_seed, condition=self.condition, epoch_index=epoch_index,
                        seed_context=self.seed_context,
                    )
                    if self.condition != "D"
                    else None
                ),
                "outcome_counts": counts.tolist(),
                "configured_success_probability": self.success_probability,
                "configured_success_probability_by_action": self.success_probability_by_action,
                "configured_success_probability_by_action_and_size": (
                    self.success_probability_by_action_and_size
                ),
                "configured_success_probability_by_action_and_shape": (
                    self.success_probability_by_action_and_shape
                ),
                "epoch_empirical_success_probability": (
                    float(counts[1].item() / self.samples_per_epoch)
                    if self.condition == "S"
                    else None
                ),
                "ordered_outcome_indices_sha256": hashlib.sha256(
                    outcome_indices.numpy().tobytes()
                ).hexdigest(),
                "local_ordered_outcome_indices_sha256": hashlib.sha256(
                    local_outcome_indices.numpy().tobytes()
                ).hexdigest(),
            }
        )
        if self.seed_context is not None:
            metadata["outcome_stream_key"] = (
                self.seed_context.key("data-outcome", partition=self.condition, epoch=epoch_index).record()
                if self.condition != "D" else None
            )
            metadata["outcome_root_seed"] = None  # The legacy integer does not derive this stream.
        if self.condition == "S" and self.success_probability_by_action is not None:
            source_indices = self.source_sampler.indices_for_epoch(epoch_index)
            action_statistics = {}
            for action_index, action_name in enumerate(MPI3D_CARDINAL_ACTION_NAMES):
                selected = source_indices.remainder(len(MPI3D_CARDINAL_ACTION_NAMES)) == action_index
                action_outcomes = outcome_indices[selected]
                action_statistics[action_name] = {
                    "count": int(action_outcomes.numel()),
                    "success_count": int(action_outcomes.sum().item()),
                    "empirical_success_probability": float(action_outcomes.float().mean().item()),
                }
            metadata["action_outcome_statistics"] = action_statistics
        state_table, state_level_names, source_state_ids = (
            self.active_state_action_probabilities()
        )
        if self.condition == "S" and state_table is not None:
            assert source_state_ids is not None
            statistics_key = (
                "size_action_outcome_statistics"
                if self.success_probability_by_action_and_size is not None
                else "shape_action_outcome_statistics"
            )
            source_indices = self.source_sampler.indices_for_epoch(epoch_index)
            sampled_state_ids = source_state_ids[source_indices]
            cell_statistics = {}
            for state_id, state_name in enumerate(state_level_names):
                cell_statistics[state_name] = {}
                for action_index, action_name in enumerate(
                    MPI3D_CARDINAL_ACTION_NAMES
                ):
                    selected = (
                        sampled_state_ids == state_id
                    ) & (
                        source_indices.remainder(len(MPI3D_CARDINAL_ACTION_NAMES))
                        == action_index
                    )
                    cell_outcomes = outcome_indices[selected]
                    cell_statistics[state_name][action_name] = {
                        "count": int(cell_outcomes.numel()),
                        "success_count": int(cell_outcomes.sum().item()),
                        "empirical_success_probability": float(
                            cell_outcomes.float().mean().item()
                        ),
                    }
            metadata[statistics_key] = cell_statistics
        return metadata

    def __iter__(self):
        return iter(self.visits_for_epoch(self.epoch_index))

    def __len__(self) -> int:
        return len(self.source_sampler)


class MPI3DDataSource:
    """Build MPI3D transition datasets while keeping N/S evaluation explicit."""

    def build_transition_dataset(
        self,
        cfg: Dict[str, object],
        split: str,
        *,
        return_label: bool = False,
        return_original: bool = False,
    ) -> Dataset:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown MPI3D split: {split}")
        if return_label or return_original:
            raise ValueError("MPI3D transition samples expose factor metadata directly")
        data_cfg = cfg["data"]
        condition = data_cfg.get("condition")
        if condition not in MPI3D_CORE_CONDITIONS:
            supported = ", ".join(MPI3D_CORE_CONDITIONS)
            raise ValueError(
                f"MPI3D requires data.condition in {{{supported}}}; got {condition!r}"
            )
        if condition in {"N", "S"} and split != "train":
            raise ValueError(
                "MPI3D N/S validation and test require repeated-future evaluation, "
                "which is not implemented yet"
            )
        images_path = data_cfg.get("images_path")
        if not isinstance(images_path, str) or not images_path:
            raise ValueError("MPI3D requires data.images_path for extracted images.npy")
        positions = None
        if split != "train":
            positions = load_mpi3d_position_manifest(
                _mpi3d_manifest_path(data_cfg, split),
                expected_name=split,
            )
        archive = MPI3DArchive(images_path)
        if condition == "D":
            return MPI3DDeterministicTransitionDataset(
                archive,
                split=split,
                positions=positions,
            )
        return MPI3DConditionTransitionDataset(
            archive,
            split=split,
            condition=condition,
            positions=positions,
        )

    def build_k_view_dataset(self, cfg: Dict[str, object], split: str) -> Dataset:
        if split not in {"validation", "test"}:
            raise ValueError("MPI3D four-action retrieval is defined only for validation and test")
        data_cfg = cfg["data"]
        if data_cfg.get("condition") != "D":
            raise ValueError("The current MPI3D data adapter authorizes only data.condition: D")
        images_path = data_cfg.get("images_path")
        if not isinstance(images_path, str) or not images_path:
            raise ValueError("MPI3D requires data.images_path for extracted images.npy")
        mrr_cfg = cfg.get("eval", {}).get("mrr")
        if not isinstance(mrr_cfg, dict) or mrr_cfg.get("num_target_views") != 4:
            raise ValueError("MPI3D D retrieval requires eval.mrr.num_target_views: 4")
        positions = load_mpi3d_position_manifest(
            _mpi3d_manifest_path(data_cfg, split),
            expected_name=split,
        )
        return MPI3DDeterministicKViewDataset(
            MPI3DArchive(images_path),
            split=split,
            positions=positions,
        )

    def build_repeated_future_dataset(self, cfg: Dict[str, object], split: str) -> Dataset:
        """Build the fixed complete N/S future support for validation or test."""
        if split not in {"validation", "test"}:
            raise ValueError("MPI3D repeated-future evaluation is defined only for validation and test")
        data_cfg = cfg["data"]
        condition = data_cfg.get("condition")
        if condition not in {"N", "S"}:
            raise ValueError("MPI3D repeated-future evaluation requires data.condition: N or S")
        images_path = data_cfg.get("images_path")
        if not isinstance(images_path, str) or not images_path:
            raise ValueError("MPI3D requires data.images_path for extracted images.npy")
        positions = load_mpi3d_position_manifest(
            _mpi3d_manifest_path(data_cfg, split),
            expected_name=split,
        )
        return MPI3DRepeatedFutureDataset(
            MPI3DArchive(images_path),
            split=split,
            condition=condition,
            positions=positions,
        )

    def build_clean_labeled_dataset(self, cfg: Dict[str, object], split: str) -> Dataset:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown MPI3D clean-image split: {split}")
        data_cfg = cfg["data"]
        if data_cfg.get("condition") not in MPI3D_CORE_CONDITIONS:
            supported = ", ".join(MPI3D_CORE_CONDITIONS)
            raise ValueError(
                f"MPI3D clean-image probes require data.condition in {{{supported}}}"
            )
        images_path = data_cfg.get("images_path")
        if not isinstance(images_path, str) or not images_path:
            raise ValueError("MPI3D requires data.images_path for extracted images.npy")
        positions = resolve_mpi3d_probe_positions(data_cfg, split=split)
        return MPI3DCleanImageDataset(
            MPI3DArchive(images_path),
            split=split,
            positions=positions,
        )


def generate_balanced_mpi3d_position_manifest(
    *,
    num_positions: int,
    generator: random.Random,
    excluded_positions: Iterable[Sequence[int]] = (),
) -> Tuple[Tuple[int, int], ...]:
    """Generate one balanced manifest, optionally excluding complete position pairs.

    One round pairs every legal horizontal value with a permutation of all
    legal vertical values. A randomized bipartite matching keeps every round
    balanced even when many pairs are reserved by an earlier split.
    """
    legal_values = tuple(range(MPI3D_COMMON_SOURCE_MIN, MPI3D_COMMON_SOURCE_MAX + 1))
    if num_positions <= 0 or num_positions % len(legal_values) != 0:
        raise ValueError(
            "num_positions must be a positive multiple of the 32 legal MPI3D axis values"
        )

    seen_positions = set(
        _validate_mpi3d_position_pair(position) for position in excluded_positions
    )
    positions = []
    for _ in range(num_positions // len(legal_values)):
        horizontal_order = list(legal_values)
        generator.shuffle(horizontal_order)
        candidates = {}
        for horizontal_axis in horizontal_order:
            available = [
                vertical_axis
                for vertical_axis in legal_values
                if (horizontal_axis, vertical_axis) not in seen_positions
            ]
            generator.shuffle(available)
            candidates[horizontal_axis] = available

        matched_horizontal_by_vertical = {}

        def assign(horizontal_axis: int, visited_verticals: set[int]) -> bool:
            for vertical_axis in candidates[horizontal_axis]:
                if vertical_axis in visited_verticals:
                    continue
                visited_verticals.add(vertical_axis)
                previous_horizontal = matched_horizontal_by_vertical.get(vertical_axis)
                if previous_horizontal is None or assign(previous_horizontal, visited_verticals):
                    matched_horizontal_by_vertical[vertical_axis] = horizontal_axis
                    return True
            return False

        if not all(assign(horizontal_axis, set()) for horizontal_axis in horizontal_order):
            raise ValueError("excluded MPI3D position pairs leave no balanced manifest round")
        matched_vertical_by_horizontal = {
            horizontal_axis: vertical_axis
            for vertical_axis, horizontal_axis in matched_horizontal_by_vertical.items()
        }
        round_positions = tuple(
            (horizontal_axis, matched_vertical_by_horizontal[horizontal_axis])
            for horizontal_axis in legal_values
        )
        positions.extend(round_positions)
        seen_positions.update(round_positions)
    return tuple(positions)


def resolve_mpi3d_probe_positions(
    data_cfg: Dict[str, object],
    *,
    split: str,
) -> Tuple[Tuple[int, int], ...]:
    """Resolve the standard probe manifests or the pair-disjoint robustness split.

    The robustness protocol leaves the frozen encoder and the original
    probe-train manifest unchanged. Validation and test are deterministic,
    axis-balanced sets of complete ``(horizontal, vertical)`` pairs that are
    mutually disjoint from probe train and from one another.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown MPI3D clean-image split: {split}")
    manifest_name = "probe_train" if split == "train" else split
    probe_train = load_mpi3d_position_manifest(
        _mpi3d_manifest_path(data_cfg, "probe_train"),
        expected_name="probe_train",
    )
    partition_cfg = data_cfg.get("probe_position_partition")
    if partition_cfg is None:
        if split == "train":
            return probe_train
        return load_mpi3d_position_manifest(
            _mpi3d_manifest_path(data_cfg, manifest_name),
            expected_name=manifest_name,
        )

    expected_partition = {
        "kind": MPI3D_PROBE_PAIR_HOLDOUT_KIND,
        "root_seed": MPI3D_PROBE_PAIR_HOLDOUT_ROOT_SEED,
    }
    if partition_cfg != expected_partition:
        raise ValueError(
            "MPI3D probe position robustness requires data.probe_position_partition: "
            f"{expected_partition!r}"
        )
    if split == "train":
        return probe_train

    validation = generate_balanced_mpi3d_position_manifest(
        num_positions=MPI3D_POSITION_MANIFEST_SPECS["validation"],
        generator=random.Random(MPI3D_PROBE_PAIR_HOLDOUT_ROOT_SEED),
        excluded_positions=probe_train,
    )
    if split == "validation":
        return validation
    return generate_balanced_mpi3d_position_manifest(
        num_positions=MPI3D_POSITION_MANIFEST_SPECS["test"],
        generator=random.Random(MPI3D_PROBE_PAIR_HOLDOUT_ROOT_SEED + 1),
        excluded_positions=probe_train + validation,
    )


def validate_mpi3d_position_manifest(
    positions: Iterable[Sequence[int]],
    *,
    expected_num_positions: int,
) -> Tuple[Tuple[int, int], ...]:
    """Validate locked-manifest size, legality, uniqueness, and axis balance."""
    if expected_num_positions not in MPI3D_POSITION_MANIFEST_SPECS.values():
        raise ValueError("expected_num_positions must be one of the locked MPI3D manifest sizes")
    normalized_positions = tuple(_validate_mpi3d_position_pair(position) for position in positions)
    if len(normalized_positions) != expected_num_positions:
        raise ValueError(
            f"expected {expected_num_positions} MPI3D manifest positions, got {len(normalized_positions)}"
        )
    if len(set(normalized_positions)) != len(normalized_positions):
        raise ValueError("MPI3D position manifest must not contain duplicate position pairs")

    expected_count_per_axis_value = expected_num_positions // 32
    for axis_index, axis_name in enumerate(("horizontal", "vertical")):
        counts = {
            value: sum(position[axis_index] == value for position in normalized_positions)
            for value in range(MPI3D_COMMON_SOURCE_MIN, MPI3D_COMMON_SOURCE_MAX + 1)
        }
        if set(counts.values()) != {expected_count_per_axis_value}:
            raise ValueError(
                f"MPI3D position manifest must contain every {axis_name} value exactly "
                f"{expected_count_per_axis_value} times"
            )
    return normalized_positions


def mpi3d_position_manifest_sha256(positions: Iterable[Sequence[int]]) -> str:
    """Hash the canonical position-pair sequence stored in one manifest file."""
    normalized_positions = tuple(_validate_mpi3d_position_pair(position) for position in positions)
    canonical_json = json.dumps(normalized_positions, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def load_mpi3d_position_manifest(
    path: Union[Path, str],
    *,
    expected_name: str,
) -> Tuple[Tuple[int, int], ...]:
    """Load one stored locked manifest and verify its metadata and hash."""
    if expected_name not in MPI3D_POSITION_MANIFEST_SPECS:
        supported = ", ".join(MPI3D_POSITION_MANIFEST_SPECS)
        raise ValueError(f"Unknown MPI3D manifest name {expected_name!r}; supported: {supported}")
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MPI3D position manifest does not exist: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as manifest_file:
        payload = json.load(manifest_file)
    expected_fields = {
        "name",
        "root_seed",
        "stream_seed",
        "num_positions",
        "positions",
        "positions_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("MPI3D position manifest has unexpected fields")
    if payload["name"] != expected_name:
        raise ValueError("MPI3D position manifest name does not match the requested manifest")
    if payload["root_seed"] != MPI3D_MANIFEST_ROOT_SEED:
        raise ValueError("MPI3D position manifest has an unexpected root seed")
    if payload["stream_seed"] != MPI3D_POSITION_MANIFEST_STREAM_SEEDS[expected_name]:
        raise ValueError("MPI3D position manifest has an unexpected stream seed")
    expected_num_positions = MPI3D_POSITION_MANIFEST_SPECS[expected_name]
    if payload["num_positions"] != expected_num_positions:
        raise ValueError("MPI3D position manifest has an unexpected position count")
    if not isinstance(payload["positions"], list):
        raise ValueError("MPI3D position manifest positions must be a JSON list")
    positions = validate_mpi3d_position_manifest(
        payload["positions"],
        expected_num_positions=expected_num_positions,
    )
    if payload["positions_sha256"] != mpi3d_position_manifest_sha256(positions):
        raise ValueError("MPI3D position manifest SHA-256 does not match its stored positions")
    return positions


def _validate_mpi3d_position_pair(position: Sequence[int]) -> Tuple[int, int]:
    if len(position) != 2:
        raise ValueError("each MPI3D position manifest item must be [horizontal, vertical]")
    horizontal_axis, vertical_axis = position
    _validate_factor_id("horizontal_axis", horizontal_axis, MPI3D_FACTOR_SIZES[5])
    _validate_factor_id("vertical_axis", vertical_axis, MPI3D_FACTOR_SIZES[6])
    if not MPI3D_COMMON_SOURCE_MIN <= horizontal_axis <= MPI3D_COMMON_SOURCE_MAX:
        raise ValueError("MPI3D manifest horizontal_axis is outside the common legal source region")
    if not MPI3D_COMMON_SOURCE_MIN <= vertical_axis <= MPI3D_COMMON_SOURCE_MAX:
        raise ValueError("MPI3D manifest vertical_axis is outside the common legal source region")
    return horizontal_axis, vertical_axis


def _all_mpi3d_common_legal_positions() -> Tuple[Tuple[int, int], ...]:
    return tuple(
        (horizontal_axis, vertical_axis)
        for horizontal_axis in range(MPI3D_COMMON_SOURCE_MIN, MPI3D_COMMON_SOURCE_MAX + 1)
        for vertical_axis in range(MPI3D_COMMON_SOURCE_MIN, MPI3D_COMMON_SOURCE_MAX + 1)
    )


def _validate_unique_legal_mpi3d_positions(
    positions: Iterable[Sequence[int]],
) -> Tuple[Tuple[int, int], ...]:
    normalized_positions = tuple(_validate_mpi3d_position_pair(position) for position in positions)
    if not normalized_positions:
        raise ValueError("MPI3D transition positions cannot be empty")
    if len(set(normalized_positions)) != len(normalized_positions):
        raise ValueError("MPI3D transition positions must not contain duplicate position pairs")
    return normalized_positions


def _mpi3d_manifest_path(data_cfg: Dict[str, object], manifest_name: str) -> str:
    manifest_paths = data_cfg.get("position_manifest_paths")
    if not isinstance(manifest_paths, dict):
        raise ValueError("MPI3D requires data.position_manifest_paths")
    manifest_path = manifest_paths.get(manifest_name)
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ValueError(f"MPI3D requires a path for the {manifest_name!r} position manifest")
    return manifest_path


def _validate_mpi3d_images_array(images: object) -> None:
    if getattr(images, "shape", None) != MPI3D_IMAGES_ARRAY_SHAPE:
        raise ValueError(
            "MPI3D images array must have shape "
            f"{MPI3D_IMAGES_ARRAY_SHAPE}, got {getattr(images, 'shape', None)}"
        )
    if getattr(images, "dtype", None) != np.dtype(np.uint8):
        raise ValueError("MPI3D images array must use uint8 RGB values")


def _validate_factor_id(factor_name: str, factor_id: int, factor_size: int) -> None:
    if isinstance(factor_id, bool) or not isinstance(factor_id, int):
        raise TypeError(f"{factor_name} must be an integer factor ID")
    if not 0 <= factor_id < factor_size:
        raise ValueError(f"{factor_name} must be in [0, {factor_size}), got {factor_id}")
