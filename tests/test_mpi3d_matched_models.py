"""Matched MPI3D role, information, gradient, and exact restart contracts."""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from iwm_replication.distributed import DistributedContext
from iwm_replication.moving_mnist_evaluation import state_hash
from iwm_replication.mpi3d_byol import (
    MATCHED_ROLES, MATCHED_VERSION, MPI3DGlobal, build_model, file_hash,
    resolve_matched_config, validate_config,
)
from iwm_replication.paired_initialization import initialize_paired_model
from iwm_replication.seed_streams import SeedContext
from iwm_replication.utils import load_yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_mpi3d_byol as training


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(role="S1", *, tiny=True, replication=0, purpose="software-smoke"):
    base = load_yaml("config/campaigns/five_seed_v1/mpi3d.yaml")
    if tiny:
        base["device"] = "cpu"
        base["data"].update(image_size=16, batch_size=4, num_workers=0)
        base["model"].update(vit_dim=16, vit_depth=1, vit_heads=2,
                             projector_dim=8, head_hidden_dim=16, patch_size=4)
        base["train"].update(epochs=2, warmup_epochs=0, transition_samples_per_epoch=8, checkpoint_every=1)
    return resolve_matched_config(base, role=role, replication=replication, purpose=purpose)


def model(role="S1"):
    cfg = config(role)
    result = build_model(cfg)
    initialize_paired_model(result, SeedContext.from_dict(cfg["seed_streams"]))
    return result


def batch():
    generator = torch.Generator().manual_seed(32)
    return {"x_source": torch.rand(4, 3, 16, 16, generator=generator),
            "y_target": torch.rand(4, 3, 16, 16, generator=generator),
            "action": torch.tensor([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])}


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_all_four_initializations_pair_and_no_shared_target_duplicate():
    fingerprints = {}
    states = {}
    context = SeedContext("mpi3d", "software-smoke", 0)
    for index, role in enumerate(MATCHED_ROLES):
        torch.manual_seed(33 + index)
        instance = build_model(config(role))
        provenance = initialize_paired_model(instance, context)
        fingerprints[role] = provenance["tensor_hashes"]
        states[role] = instance.get_extra_state()
        if role.startswith("R"):
            assert_nested_equal(instance.online.state_dict(), instance.target.state_dict())
            assert all(not p.requires_grad for p in instance.target.parameters())
        else:
            assert instance.target_branch is instance.online
            assert not hasattr(instance, "target")
            assert not hasattr(instance, "ema_updates")
            assert not any(key.startswith("target.") for key in instance.state_dict())
    for role in MATCHED_ROLES:
        for name, value in fingerprints["R0"].items():
            assert fingerprints[role][name] == value
    assert fingerprints["R1"] == fingerprints["S1"]
    assert all(torch.equal(a, b) for a, b in zip(states["S0"]["sigreg_rng"].values(), states["S1"]["sigreg_rng"].values()))
    assert not torch.equal(*states["S0"]["sigreg_rng"].values())


@pytest.mark.parametrize("role", MATCHED_ROLES)
def test_objective_information_gradient_and_bn_contract(role):
    instance = model(role)
    data = batch()
    source, target = [data[key].requires_grad_() for key in ("x_source", "y_target")]
    features, predictions, gaussian = [], [], {}
    handles = [instance.online.register_forward_hook(lambda m, i, o: features.append(o["projector"])),
               instance.predictor.register_forward_hook(lambda m, i, o: predictions.append((i[0], o)))]
    target_features = []
    if instance.uses_ema:
        handles.append(instance.target.register_forward_hook(lambda m, i, o: target_features.append(o["projector"])))
    if instance.is_stochastic:
        handles.append(instance.prior.register_forward_hook(lambda m, i, o: gaussian.update(prior=o, prior_input=i[0])))
        handles.append(instance.posterior.register_forward_hook(lambda m, i, o: gaussian.update(posterior=o, posterior_input=i[0])))
    result = instance(source, target, data["action"], generator=torch.Generator().manual_seed(81))
    for handle in handles:
        handle.remove()
    true_target = target_features[0] if instance.uses_ema else features[1]
    expected = -torch.nn.functional.cosine_similarity(predictions[0][1], true_target, dim=-1).mean()
    torch.testing.assert_close(result["alignment"], expected)
    torch.testing.assert_close(result["loss"], expected + instance.beta * result["kl"] + instance.sigreg_weight * result["sigreg"])
    torch.testing.assert_close(result["sigreg"], .5 * (result["source_sigreg"] + result["target_sigreg"]))
    assert true_target.requires_grad == (not instance.uses_ema)
    assert len(features) == (1 if role == "R0" else 2)
    if instance.is_stochastic:
        pm, ps = gaussian["prior"]
        qm, qs = gaussian["posterior"]
        torch.testing.assert_close(result["kl"], kl_divergence(Normal(qm, qs.exp()), Normal(pm, ps.exp())).sum(-1).mean())
        assert gaussian["prior_input"].shape[-1] == 10
        assert gaussian["posterior_input"].shape[-1] == 18
        assert torch.equal(gaussian["posterior_input"][:, -8:], features[1])
        assert torch.equal(gaussian["prior_input"][:, -2:], data["action"])
        prior_gradient = torch.autograd.grad(result["alignment"], tuple(instance.prior.parameters()),
                                             allow_unused=True, retain_graph=True)
        assert all(value is None for value in prior_gradient)
    else:
        assert result["kl"].item() == 0
        assert torch.count_nonzero(predictions[0][0][:, -1]) == 0
    result["loss"].backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    assert (target.grad is None) == (role == "R0")
    if target.grad is not None:
        assert target.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in instance.parameters() if p.requires_grad)
    if instance.uses_ema:
        assert all(p.grad is None for p in instance.target.parameters())
    assert instance.online.projector[1].num_batches_tracked.item() == (1 if role == "R0" else 2)
    if instance.uses_ema:
        assert instance.target.projector[1].num_batches_tracked.item() == 1
    with pytest.raises(TypeError):
        instance(source, target, data["action"], execution_success=torch.ones(4))


@pytest.mark.parametrize("role", MATCHED_ROLES)
def test_optimizer_counter_and_parameter_only_ema(role):
    instance = model(role)
    old = deepcopy(instance.target.state_dict()) if instance.uses_ema else None
    with torch.no_grad():
        for parameter in instance.online.parameters():
            parameter.add_(.3)
        instance.online.projector[1].running_mean.add_(4)
    instance.after_optimizer_step(.8)
    assert instance.optimizer_updates.item() == 1
    if instance.uses_ema:
        assert instance.ema_updates.item() == 1
        for name, parameter in instance.target.named_parameters():
            torch.testing.assert_close(parameter, .8 * old[name] + .2 * instance.online.get_parameter(name))
        assert torch.equal(instance.target.projector[1].running_mean, old["projector.1.running_mean"])
    else:
        assert not hasattr(instance, "ema_updates")


@pytest.mark.parametrize("role", MATCHED_ROLES)
def test_frozen_prior_only_forecast_raw_normalized_and_prior_mean(role, monkeypatch):
    instance = model(role).eval().requires_grad_(False)
    data = batch()
    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden future or posterior access")
    if instance.uses_ema:
        monkeypatch.setattr(instance.target, "forward", forbidden)
    if instance.is_stochastic:
        monkeypatch.setattr(instance.posterior, "forward", forbidden)
    before = state_hash(instance)
    raw, weights = instance.forecast(data["x_source"], data["action"], quantiles=4, normalize=False)
    normalized, w = instance.forecast(data["x_source"], data["action"], quantiles=4)
    torch.testing.assert_close(normalized, torch.nn.functional.normalize(raw, dim=-1))
    assert torch.equal(weights, w)
    assert raw.shape == (4, 4 if instance.is_stochastic else 1, 8)
    torch.testing.assert_close(weights.sum(-1), torch.ones(4))
    mean, mw = instance.forecast(data["x_source"], data["action"], fixed_residual=True, normalize=False)
    assert mean.shape == (4, 1, 8) and torch.equal(mw, torch.ones(4, 1))
    if not instance.is_stochastic:
        assert torch.equal(raw, mean)
    assert state_hash(instance) == before
    instance.train()
    with pytest.raises(RuntimeError, match="frozen evaluation"):
        instance.forecast(data["x_source"], data["action"])


def test_five_replications_cli_mapping_and_recipe_rejection():
    seen = set()
    parser = training.argument_parser()
    for role in MATCHED_ROLES:
        for replication in range(1, 6):
            cfg = training.config_from_arguments(parser.parse_args([
                "--config", "config/campaigns/five_seed_v1/mpi3d.yaml", "--role", role,
                "--replication", str(replication)]))
            assert cfg["role"] == role and cfg["seed"] == replication
            assert cfg["protocol"] == MATCHED_VERSION
            seen.add(cfg["run_name"])
    assert len(seen) == 20
    for index in range(20):
        cfg = training.config_from_arguments(parser.parse_args([
            "--config", "config/campaigns/five_seed_v1/mpi3d.yaml", "--array-index", str(index)]))
        assert cfg["role"] == MATCHED_ROLES[index // 5] and cfg["seed"] == index % 5 + 1
    full = config("R0", tiny=False, replication=5, purpose="main-training")
    for key, value in (("seed", 6), ("target_bn", "independent_batch_stats_parameter_only_ema")):
        broken = deepcopy(full)
        broken[key] = value
        with pytest.raises(ValueError):
            validate_config(broken)
    broken = deepcopy(full)
    broken["loss"]["beta"] = .01
    with pytest.raises(ValueError, match="recipe changed"):
        validate_config(broken)
    broken = deepcopy(full)
    broken["campaign_manifest"]["seed_manifest_sha256"] = "tampered"
    with pytest.raises(ValueError, match="binding mismatch"):
        validate_config(broken)


@pytest.mark.parametrize("role", MATCHED_ROLES)
def test_actual_trainer_epoch_resume_exact_including_sigreg(tmp_path, monkeypatch, role):
    cfg = config(role)
    cfg["output_dir"] = str(tmp_path)
    fake_file = tmp_path / "synthetic-archive"
    fake_file.write_bytes(b"synthetic fixture; no real MPI3D data")
    cfg["data"].update(images_path=str(fake_file), images_sha256=file_hash(fake_file))
    images = torch.rand(32, 3, 16, 16, generator=torch.Generator().manual_seed(101))
    class SyntheticPopulation:
        def __len__(self):
            return len(images)
        def __getitem__(self, visit):
            index, outcome = visit.source_index, visit.outcome_index
            return {"x_source": images[index], "y_target": images[(index + outcome) % len(images)],
                    "action": torch.tensor(((1., 0.), (-1., 0.), (0., 1.), (0., -1.)))[index % 4],
                    "outcome_label": 999}
    monkeypatch.setattr(training, "build_iwm_dataset", lambda cfg, split: SyntheticPopulation())
    # A fixed synthetic source artifact avoids unrelated concurrent workspace
    # edits changing the test's actual checkpoint provenance during its two runs.
    monkeypatch.setattr(training, "provenance", lambda cfg: {str(fake_file): file_hash(fake_file)})
    monkeypatch.setattr(training, "tee_console_to_file", lambda path: None)
    topology = DistributedContext(0, 1, 0, torch.device("cpu"))
    args = SimpleNamespace(preflight=False, synthetic_preflight=False, resume=False, stop_after_epoch=1)
    training.run(args, cfg, topology)
    checkpoint_dir = tmp_path / cfg["run_name"] / "checkpoints"
    paused = torch.load(checkpoint_dir / "checkpoint_latest.pt", weights_only=True)
    assert paused["epoch"] == 1 and paused["config"]["train"]["epochs"] == 2
    assert json.loads((checkpoint_dir.parent / "metrics/training.json").read_text())["status"] == "training-paused"
    assert not (checkpoint_dir / "checkpoint_epoch_0002.pt").exists()
    uninterrupted_cfg = deepcopy(cfg)
    uninterrupted_cfg["output_dir"] = str(tmp_path / "uninterrupted")
    training.run(SimpleNamespace(preflight=False, synthetic_preflight=False, resume=False), uninterrupted_cfg, topology)
    uninterrupted = torch.load(Path(uninterrupted_cfg["output_dir"]) / cfg["run_name"] /
                               "checkpoints/checkpoint_latest.pt", weights_only=True)
    args.resume, args.stop_after_epoch = True, None
    training.run(args, cfg, topology)
    checkpoint_dir = tmp_path / cfg["run_name"] / "checkpoints"
    completed = torch.load(checkpoint_dir / "checkpoint_latest.pt", weights_only=True)
    for field in ("model", "optimizer", "rng_by_rank", "history", "global_step",
                  "sampler_metadata", "initialization"):
        assert_nested_equal(completed[field], uninterrupted[field])
    assert completed["version"] == MATCHED_VERSION and completed["role"] == role
    assert completed["model"]["optimizer_updates"].item() == 4
    assert len(completed["model"]["_extra_state"]["sigreg_rng"]) == (0 if role.startswith("R") else 2)
    metadata = json.loads((checkpoint_dir.parent / "metadata.json").read_text())
    assert metadata["batchnorm_passes"] == {"online": 1 if role == "R0" else 2, "ema": int(role.startswith("R"))}
    (checkpoint_dir / "checkpoint_latest.pt").write_bytes((checkpoint_dir / "checkpoint_epoch_0001.pt").read_bytes())
    args.resume = True
    random.random(); np.random.rand(11); torch.rand(23)
    training.run(args, cfg, topology)
    resumed = torch.load(checkpoint_dir / "checkpoint_latest.pt", weights_only=True)
    assert_nested_equal(completed, resumed)
    resumed["model"]["optimizer_updates"].add_(1)
    torch.save(resumed, checkpoint_dir / "checkpoint_latest.pt")
    with pytest.raises(ValueError, match="counters disagree"):
        training.run(args, cfg, topology)


def _global_sigreg_gradient(rank, init_file, output_file):
    import torch.distributed as dist
    from iwm_replication.lewm_adassl import PatchSIGReg
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        weight, source, target = _sigreg_fixture()
        context = SeedContext("mpi3d", "software-smoke", 0)
        penalty = PatchSIGReg(global_batch_size=4, num_projections=128, knots=17,
                             interval=(.2, 4.), rng_seed=0)
        loss = sum(penalty((view[rank::2] @ weight)[:, None],
                   generator=context.torch_generator("sigreg", view=name)).sigreg
                   for name, view in (("source", source), ("target", target))) * .05
        loss.backward()
        dist.all_reduce(weight.grad)
        weight.grad.div_(2)  # Same averaging performed by DDP.
        if rank == 0:
            torch.save(weight.grad, output_file)
    finally:
        dist.destroy_process_group()


def _sigreg_fixture():
    generator = torch.Generator().manual_seed(18)
    return (torch.randn(3, 8, generator=generator).requires_grad_(),
            torch.randn(4, 3, generator=generator), torch.randn(4, 3, generator=generator))


def test_singleton_global_sigreg_ddp_gradient_matches_full_batch(tmp_path):
    import torch.multiprocessing as mp
    from iwm_replication.lewm_adassl import PatchSIGReg
    output = tmp_path / "sigreg-gradient.pt"
    mp.spawn(_global_sigreg_gradient, args=(str(tmp_path / "gloo-init"), str(output)),
             nprocs=2, join=True)
    weight, source, target = _sigreg_fixture()
    context = SeedContext("mpi3d", "software-smoke", 0)
    penalty = PatchSIGReg(global_batch_size=4, num_projections=128, knots=17,
                         interval=(.2, 4.), rng_seed=0)
    loss = sum(penalty((view @ weight)[:, None],
               generator=context.torch_generator("sigreg", view=name)).sigreg
               for name, view in (("source", source), ("target", target))) * .05
    loss.backward()
    torch.testing.assert_close(torch.load(output, weights_only=True), weight.grad, rtol=1e-5, atol=1e-6)
