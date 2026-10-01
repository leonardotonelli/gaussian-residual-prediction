"""Frozen campaign readout, scoring and artifact contracts shared by both datasets.

Transforms are selected using real selection features only. Final evaluation
loads the fitted development artifact; no target score can select a head.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .seed_streams import preserve_rng

EVALUATION_VERSION = 'five-seed-frozen-evaluation-v1'


def load_evaluation_config(path):
    """Read generated JSON without YAML 1.1 reinterpreting exponent floats."""
    path = Path(path)
    if path.suffix.lower() == '.json':
        config = json.loads(path.read_text())
    else:
        import yaml
        config = yaml.safe_load(path.read_text())
    validate_evaluation_config(config)
    return config


def feature_transform(x, name):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite feature rows')
    if name == 'raw':
        return x
    if name == 'unit':
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    raise ValueError('Feature transform must be raw or unit')


def fit_readout_candidates(x, y, selection_x, selection_y, alphas, *, transforms=('raw', 'unit')):
    from .moving_mnist_evaluation import fit_ridge
    if not transforms or len(set(transforms)) != len(transforms) or any(t not in ('raw', 'unit') for t in transforms):
        raise ValueError('Declare unique raw/unit readout candidates')
    candidates, heads = {}, {}
    for mode in transforms:
        head, details = fit_ridge(feature_transform(x, mode), np.asarray(y),
                                 feature_transform(selection_x, mode), np.asarray(selection_y), alphas)
        head['feature_transform'] = mode
        head['train_standardized_norm_p99'] = details['train_standardized_norm_p99']
        candidates[mode], heads[mode] = details, head
    # A fixed raw-first tie break prevents selection by model forecast scores.
    selected = min(candidates, key=lambda key: (candidates[key]['selection_mse'], key != 'raw'))
    return heads[selected], {'selected_transform': selected, 'candidates': candidates,
        'selection_rule': 'minimum real-feature selection MSE; exact ties prefer raw',
        'normalizer_fit_partition': 'fit only', 'forecast_scores_used_for_selection': False}


def fit_affine_mse(x, y):
    """Converged unregularized affine MSE; expose rank instead of hiding singularity."""
    from .moving_mnist_evaluation import standardizer, transform
    x, y = np.asarray(x), np.asarray(y, dtype=np.float64)
    mean, scale, active = standardizer(x)
    head = dict(mean=mean, scale=scale, active=active, bias=y.mean(0), feature_transform='raw')
    z = transform(x, head)
    weight, _, rank, singular = np.linalg.lstsq(z, y - head['bias'], rcond=None)
    head['weight'] = weight
    threshold = max(z.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0.)
    positive = singular[singular > threshold]
    return head, {'objective': 'unregularized affine MSE with intercept; exact least-squares solve',
        'rank': int(rank), 'features': z.shape[1], 'fit_examples': len(z),
        'condition_over_resolved_subspace': float(positive[0] / positive[-1]) if len(positive) else None,
        'rank_deficient': bool(rank < z.shape[1]), 'normalizer_fit_partition': 'fit only'}


def iid_scores(predictions, truth):
    """Euclidean ES U statistic and nearest-support coverage, arbitrary dimension."""
    p, y = np.asarray(predictions, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if p.ndim != 3 or y.ndim != 3 or p.shape[0] != y.shape[0] or p.shape[-1] != y.shape[-1] or min(*p.shape, *y.shape) < 1:
        raise ValueError('Expected nonempty (queries, draws, dimensions) arrays')
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite score inputs')
    cross = np.linalg.norm(p[:, :, None] - y[:, None], axis=-1)
    energy = cross.mean((1, 2))
    if p.shape[1] > 1:
        pair = np.linalg.norm(p[:, :, None] - p[:, None], axis=-1)
        energy -= pair.sum((1, 2)) / (2 * p.shape[1] * (p.shape[1] - 1))
    return {'energy_score': energy, 'coverage_squared': np.square(cross.min(1)).mean(1)}


def assert_disjoint_partitions(partitions):
    seen = set()
    for name, ids in partitions.items():
        values = [str(value) for value in ids]
        if len(set(values)) != len(values) or seen.intersection(values):
            raise ValueError(f'Overlapping or duplicate identity/image IDs in {name}')
        seen.update(values)


def split_development_indices(size, *, seed_context, module):
    if size < 4 or seed_context.purpose != 'evaluation' or seed_context.replication is not None:
        raise ValueError('Need fixed evaluation context and at least four development rows')
    order = seed_context.numpy_rng('evaluation-bank', module=module, partition='development').permutation(size)
    return {'selection': order[:size // 2], 'report': order[size // 2:]}


def require_final_contract(mode, *, fitted_artifacts=None, fitted_sha256=None, frozen_protocol_sha256=None):
    if mode not in ('development', 'final'):
        raise ValueError('Evaluation mode must be development or final')
    if mode == 'final':
        for label, value in (('fitted artifact SHA256', fitted_sha256), ('frozen protocol SHA256', frozen_protocol_sha256)):
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'Final evaluation requires a frozen {label}')
        if fitted_artifacts is None:
            raise ValueError('Final evaluation requires fitted development artifacts')


@contextmanager
def frozen_evaluation(model):
    from .moving_mnist_evaluation import state_hash
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Evaluation requires frozen parameters and every module in eval mode')
    before = state_hash(model)
    with preserve_rng():
        try:
            yield before
        finally:
            if (before != state_hash(model) or any(m.training for m in model.modules())
                    or any(p.requires_grad for p in model.parameters())):
                raise RuntimeError("Evaluation mutated frozen parameter/buffer/RNG state or modes")


def _map_arrays(value, *, to_torch):
    if isinstance(value, dict):
        return {key: _map_arrays(item, to_torch=to_torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_map_arrays(item, to_torch=to_torch) for item in value]
    if to_torch and isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if not to_torch and torch.is_tensor(value):
        return value.cpu().numpy().copy()
    return value


def save_fitted(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open('xb') as handle:
        torch.save(_map_arrays(payload, to_torch=True), handle)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fitted(path, expected_sha256):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Fitted artifact SHA256 mismatch')
    return _map_arrays(torch.load(path, map_location='cpu', weights_only=True), to_torch=False)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    with path.open('x') as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def validate_evaluation_config(config):
    """Validate the proposed/frozen numeric protocol without touching datasets."""
    if not isinstance(config, dict) or config.get('version') != EVALUATION_VERSION:
        raise ValueError('Unknown campaign evaluation protocol')
    def count(value, label, minimum=1):
        if type(value) is not int or value < minimum:
            raise ValueError(f'{label} must be an integer >= {minimum}')
    def number(value, label, positive=False):
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f'{label} must be finite and {"positive" if positive else "nonnegative"}')
    for dataset in ('moving_mnist', 'mpi3d'):
        settings = config.get(dataset)
        if not isinstance(settings, dict):
            raise ValueError(f'Missing {dataset} evaluation settings')
        count(settings.get('batch_size'), dataset + ' batch size', 2 if dataset == 'mpi3d' else 1)
        transforms = settings.get('readout_transforms')
        if (not isinstance(transforms, list) or not transforms or len(set(transforms)) != len(transforms)
                or any(t not in ('raw', 'unit') for t in transforms)):
            raise ValueError('Declare distinct raw/unit readout candidates')
        alphas = settings.get('ridge_alphas')
        if not isinstance(alphas, list) or not alphas:
            raise ValueError('Declare nonempty readout ridge penalties')
        for alpha in alphas:
            number(alpha, 'ridge alpha', positive=True)
    mm, mpi = config['moving_mnist'], config['mpi3d']
    if not isinstance(mm.get('identity_manifest'), str) or not mm['identity_manifest']:
        raise ValueError('Moving-MNIST identity manifest path is required')
    count(mm.get('seed'), 'legacy metadata seed', 0)
    if set(mm.get('sizes', {})) != {'fit', 'selection', 'report', 'queries'}:
        raise ValueError('Declare Moving-MNIST fit/selection/report/query counts')
    for name, value in mm['sizes'].items():
        count(value, name, 2)
    for name in ('samples', 'futures', 'latent_futures'):
        count(mm.get(name), name, 2)
    if mm['latent_futures'] > mm['futures']:
        raise ValueError('Latent truth is a declared prefix of the independent physical truth bank')
    probe = mm.get('digit_probe', {})
    for name in ('epochs', 'patience', 'batch_size'):
        count(probe.get(name), 'digit probe ' + name)
    for name in ('min_delta', 'weight_decay'):
        number(probe.get(name), 'digit probe ' + name)
    number(probe.get('learning_rate'), 'digit probe learning rate', positive=True)
    if set(mpi.get('probe_counts', {})) != {'fit', 'selection', 'report'}:
        raise ValueError('Declare MPI3D fit/selection/report probe counts')
    for name, value in mpi['probe_counts'].items():
        count(value, name, 2)
    if not isinstance(mpi.get('readout_counts'), list) or len(mpi['readout_counts']) != 3:
        raise ValueError('Declare MPI3D fit/selection/report readout counts')
    for value in mpi['readout_counts']:
        count(value, 'readout count', 2)
    count(mpi.get('queries'), 'MPI3D queries', 8)
    count(mpi.get('quantiles'), 'MPI3D quadrature count', 2)
    if mpi['queries'] % 4 or mpi['queries'] % mpi['batch_size']:
        raise ValueError('MPI3D queries must contain complete four-action groups and full batches')
    probe = mpi.get('probe', {})
    for name in ('maximum_epochs', 'batch_size'):
        count(probe.get(name), 'MPI3D probe ' + name)
    number(probe.get('learning_rate'), 'MPI3D probe learning rate', positive=True)
    if not isinstance(probe.get('lambda_grid'), list) or not probe['lambda_grid']:
        raise ValueError('Declare MPI3D probe penalties')
    for value in probe['lambda_grid']:
        number(value, 'MPI3D probe penalty')


def query_regeneration_manifest(dataset, context, settings, mode, data_fingerprints):
    """Pure specification, independent of model tensors, trained seed and data reads."""
    if context.dataset != dataset or context.purpose != 'evaluation' or context.replication is not None:
        raise ValueError('Query manifests require a fixed dataset evaluation context')
    if mode not in ('development', 'final'):
        raise ValueError('Invalid query partition')
    return {'schema': 'five-seed-query-regeneration-v1', 'dataset': dataset,
        'seed_context': context.as_dict(), 'partition': mode, 'settings': settings,
        'data_fingerprints': data_fingerprints,
        'generation': ('online-clip-addresses-and-independent-truth-oracle-v1' if dataset == 'moving_mnist'
                       else 'canonical-source-groups-pcg64-four-actions-exact-balanced-truth-v1')}


def validate_final_bindings(mode, config, dataset, query_manifest, *, analysis_contract=None,
                            frozen_protocol_sha256=None):
    """Check all freeze bindings before any final images or labels are loaded."""
    from .seed_streams import content_sha256
    if mode != 'final':
        return
    from .five_seed_analysis import validate_analysis_contract
    if analysis_contract is None:
        raise ValueError('Final evaluation requires the frozen analysis contract')
    if isinstance(analysis_contract, (str, Path)):
        import yaml
        analysis_contract = yaml.safe_load(Path(analysis_contract).read_text())
    validate_analysis_contract(analysis_contract)
    if analysis_contract['protocol_sha256'] != frozen_protocol_sha256:
        raise ValueError('Frozen protocol binding mismatch')
    if analysis_contract['evaluation_contract_sha256'] != content_sha256(config):
        raise ValueError('Frozen evaluation configuration binding mismatch')
    if analysis_contract['final_bank_sha256'][dataset] != content_sha256(query_manifest):
        raise ValueError('Frozen final query regeneration manifest mismatch')


def runtime_metadata(device):
    import platform
    result = {'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__),
              'device': str(device), 'cuda_build': torch.version.cuda}
    if device.type == 'cuda':
        result['gpu'] = torch.cuda.get_device_name(device)
    return result
