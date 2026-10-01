"""Frozen development/final Moving-MNIST campaign evaluation; never train or submit jobs."""
import argparse
from pathlib import Path

import torch
from src.campaign_evaluation import load_evaluation_config

from src.campaign_moving_mnist_evaluation import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/campaigns/five_seed_v1/evaluation.yaml'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('development', 'final'), default='development')
    parser.add_argument('--fitted-artifacts', type=Path)
    parser.add_argument('--fitted-sha256')
    parser.add_argument('--frozen-protocol-sha256')
    parser.add_argument('--analysis-contract', type=Path)
    parser.add_argument('--allow-partial-software-smoke', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    run(load_evaluation_config(args.config), checkpoint=args.checkpoint, expected_sha256=args.expected_sha256,
        data_dir=args.data_dir, output_dir=args.output, device=torch.device(args.device), mode=args.mode,
        fitted_artifacts=args.fitted_artifacts, fitted_sha256=args.fitted_sha256,
        frozen_protocol_sha256=args.frozen_protocol_sha256, analysis_contract=args.analysis_contract, allow_partial_software_smoke=args.allow_partial_software_smoke)


if __name__ == '__main__':
    main()
