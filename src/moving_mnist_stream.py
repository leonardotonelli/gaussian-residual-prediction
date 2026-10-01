"""Index-addressed online clips; reproducible across worker counts and resumes."""

from __future__ import annotations

from dataclasses import asdict
import hashlib

import numpy as np
from torch.utils.data import Dataset

from .moving_mnist import GeneratorConfig, random_stream, render_video, sample_future_velocities, sample_source
from .moving_mnist_data import content_hash
from .seed_streams import SeedContext


STREAM_VERSION = "concept2-online-clips-v1"
SEEDED_STREAM_VERSION = "concept2-online-clips-structured-v2"


class OnlineClips(Dataset):
    """Each integer address generates one image identity, motion and future.

    Training and development use distinct identities AND separate RNG streams.
    A DataLoader samples consecutive addresses; no epoch resets, mutable RNG,
    or worker seed can silently repeat/skip clips when restarting.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, ids: list[str], *,
                 split: str, seed: int, generator_config: GeneratorConfig,
                 identity_hash: str, size: int, seed_context: SeedContext | None = None,
                 seed_partition: str | None = None):
        if split not in ("train", "development"):
            raise ValueError("Training infrastructure cannot access final-test clips")
        if size < 1 or len(images) < 1 or len(images) != len(labels) or len(images) != len(ids):
            raise ValueError("Nonempty matching images/labels/identities and positive size required")
        if seed_context is not None and seed_context.dataset != "moving_mnist":
            raise ValueError("OnlineClips requires a Moving-MNIST seed context")
        self.seed_context = seed_context
        self.seed_partition = split if seed_partition is None else seed_partition
        self.images, self.labels, self.ids = images, labels, ids
        self.split, self.seed, self.config, self.size = split, seed, generator_config, size
        self.source_stream = 10 if split == "train" else 20
        self.future_stream = self.source_stream + 1
        self.contract = {"version": STREAM_VERSION, "split": split, "seed": seed,
                         "generator": asdict(generator_config), "size": size,
                         "identity_manifest_sha256": identity_hash,
                         "image_ids_sha256": content_hash({"ids": ids}),
                         "images_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
                         "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                         "source_stream": self.source_stream, "future_stream": self.future_stream}
        if seed_context is not None:
            self.contract.update(version=SEEDED_STREAM_VERSION, seed_context=seed_context.as_dict(),
                                 seed_partition=self.seed_partition,
                                 source_stream="data-source-motion", future_stream="data-future",
                                 identity_stream="data-identity")

    def __len__(self):
        return self.size

    def record(self, index: int) -> dict:
        if not 0 <= index < self.size:
            raise IndexError(index)
        if self.seed_context is None:
            rng = random_stream(self.seed, index, self.source_stream)
            identity_rng = rng
            future_rng = random_stream(self.seed, index, self.future_stream)
        else:
            address = {"partition": self.seed_partition, "sample_index": index}
            identity_rng = self.seed_context.numpy_rng("data-identity", **address)
            rng = self.seed_context.numpy_rng("data-source-motion", **address)
            future_rng = self.seed_context.numpy_rng("data-future", **address)
        image_index = int(identity_rng.integers(len(self.ids)))
        state = sample_source(self.ids[image_index], int(self.labels[image_index]), rng, self.config)
        velocity, axis = sample_future_velocities(
            state, 1, future_rng, self.config)
        return {"sample_index": index, "image_index": image_index, "state": asdict(state),
                "future_velocity": velocity[0].tolist(), "changed_axis": int(axis[0])}

    def __getitem__(self, index: int):
        from .moving_mnist import SourceState
        record = self.record(index)
        pixels = render_video(self.images[record["image_index"]], SourceState(**record["state"]),
                              np.asarray(record["future_velocity"]), self.config)
        # No digit/motion labels accompany the world-model pixel batch.
        return {**pixels, "sample_index": index}
