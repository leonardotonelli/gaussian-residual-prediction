"""Shared manifest binding and run identities for the five-seed campaign.

This module prepares and validates metadata only. It never opens datasets,
constructs evaluation banks, submits jobs, or infers that a protocol is frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .seed_streams import CAMPAIGN, SeedContext, content_sha256

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "config/campaigns/five_seed_v1/seed_manifest.json"
MANIFEST_SHA256 = "6283520677daaa751d4f3cd612467a3eafdbe76f39d26c16a42ca0d7904d3a7d"
MANIFEST_FILE_SHA256 = "860c95ea46e1c2ee9b29b008ee4a9a867c447e3b56c95bef090a6ee62fb4490e"
ROLES = ("R0", "R1", "S0", "S1")
DATASETS = ("moving_mnist", "mpi3d")
MAIN_REPLICATIONS = (1, 2, 3, 4, 5)


def manifest_binding(context: SeedContext, manifest_path=None) -> dict:
    """Bind a supported context to the already accepted, byte-exact manifest.

    A changed manifest needs an intentional campaign/version change, not merely
    a recomputed self-reported digest. Paths are not part of portable metadata.
    """
    if not isinstance(context, SeedContext) or context.campaign != CAMPAIGN:
        raise ValueError("Expected a SeedContext for the accepted campaign")
    path = DEFAULT_MANIFEST if manifest_path is None else Path(manifest_path)
    raw = path.read_bytes()
    file_digest = hashlib.sha256(raw).hexdigest()
    if file_digest != MANIFEST_FILE_SHA256:
        raise ValueError("Seed manifest differs from the accepted file hash")
    payload = json.loads(raw)
    reported = payload.pop("sha256")
    if reported != MANIFEST_SHA256 or content_sha256(payload) != reported:
        raise ValueError("Seed manifest payload hash mismatch")
    if payload["campaign"] != context.campaign:
        raise ValueError("Manifest campaign mismatch")
    matches = [item for item in payload["contexts"] if item["context"] == context.as_dict()]
    if len(matches) != 1 or matches[0]["context_sha256"] != context.sha256:
        raise ValueError("Seed context is not uniquely bound in the fixed manifest")
    return {
        "schema": "five-seed-manifest-binding-v1",
        "campaign": CAMPAIGN,
        "seed_manifest_sha256": reported,
        "seed_manifest_file_sha256": file_digest,
        "seed_context_sha256": context.sha256,
    }


def validate_manifest_binding(binding: dict, context: SeedContext, manifest_path=None) -> dict:
    expected = manifest_binding(context, manifest_path)
    if binding != expected:
        raise ValueError("Campaign seed manifest/context binding mismatch")
    return expected


def run_identity(dataset: str, role: str, replication: int, *, purpose="main-training") -> dict:
    """A role is a run identity, while its data/init context is paired by seed."""
    if dataset not in DATASETS or role not in ROLES:
        raise ValueError("Unsupported campaign dataset/role")
    if purpose not in ("main-training", "development-training", "software-smoke"):
        raise ValueError("A training run requires a training or smoke purpose")
    context = SeedContext(dataset=dataset, purpose=purpose, replication=replication)
    return {
        "run_id": f"{CAMPAIGN}/{purpose}/{dataset}/{role}/replication-{replication}",
        "dataset": dataset, "role": role, "replication": replication,
        "purpose": purpose, "seed_streams": context.as_dict(),
        "seed_context_sha256": context.sha256,
    }


def run_matrix(*, purpose="main-training", dataset=None) -> list[dict]:
    """Stable per-dataset array mapping, role-major then replication within dataset.

    This is a software mapping, not the immutable experiment registry that is
    frozen after development. No observed job status is invented here.
    """
    datasets = DATASETS if dataset is None else (dataset,)
    if any(item not in DATASETS for item in datasets):
        raise ValueError("Unsupported campaign dataset")
    if purpose not in ("main-training", "development-training", "software-smoke"):
        raise ValueError("Unsupported campaign purpose")
    replications = MAIN_REPLICATIONS if purpose == "main-training" else (0,)
    rows = [run_identity(ds, role, rep, purpose=purpose)
            for ds in datasets for role in ROLES for rep in replications]
    per_dataset = len(ROLES) * len(replications)
    return [dict(row, registry_index=index, array_index=index % per_dataset)
            for index, row in enumerate(rows)]


def run_at_index(index: int, *, dataset: str, purpose="main-training") -> dict:
    rows = run_matrix(purpose=purpose, dataset=dataset)
    if type(index) is not int or not 0 <= index < len(rows):
        raise ValueError(f"Array index must be in 0..{len(rows) - 1}")
    return rows[index]
