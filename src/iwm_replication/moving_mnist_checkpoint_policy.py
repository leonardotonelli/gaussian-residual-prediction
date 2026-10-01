"""Narrow exception for frozen checks of interrupted campaign software smokes."""
from __future__ import annotations

from .moving_mnist_campaign import CAMPAIGN_STATUS, validate_campaign_contract
from .moving_mnist_full_training import SEEDED_FULL_TRAINING_VERSION
from .seed_streams import SeedContext


def allow_partial_checkpoint(state, expected_step, role, requested=False):
    """Return whether a nonterminal checkpoint has the explicit smoke exception.

    Development/main endpoints and all legacy checkpoints retain the ordinary
    terminal-budget rule. A smoke still uses the complete 75,000-update schedule
    and batch 128, so its checkpoint is suitable for the later GPU replay check.
    """
    if type(requested) is not bool:
        raise ValueError("Partial software-smoke permission must be an explicit boolean")
    if not requested:
        return False
    context = SeedContext.from_dict(state.get("seed_context"))
    contract = state.get("contract", {})
    if (state.get("version") != SEEDED_FULL_TRAINING_VERSION
            or context.purpose != "software-smoke" or context.dataset != "moving_mnist"
            or contract.get("purpose") != CAMPAIGN_STATUS or "campaign_recipe" not in contract):
        raise ValueError("Partial frozen evaluation is restricted to campaign software-smoke checkpoints")
    validate_campaign_contract(contract)
    training = state["training_config"]
    if (type(expected_step) is not int or not 0 < expected_step <= training["total_steps"]
            or state.get("step") != expected_step
            or state.get("next_sample_index") != expected_step * training["batch_size"]
            or state.get("scheduler", {}).get("last_epoch") != expected_step):
        raise ValueError("Invalid partial software-smoke step/cursor/scheduler counters")
    if role.startswith("R") and int(state["model"]["ema_updates"]) != expected_step:
        raise ValueError("Partial software-smoke EMA count mismatch")
    for name, value in state["model"].items():
        if name.endswith("num_batches_tracked"):
            multiplier = (2 if role == "R1" and name.startswith("online_encoder.") else
                          2 if role == "S0" and name.startswith("encoder.") else
                          3 if role == "S1" and name.startswith("encoder.") else 1)
            if int(value) != multiplier * expected_step:
                raise ValueError(f"Partial software-smoke BN count mismatch: {name}")
    return True
