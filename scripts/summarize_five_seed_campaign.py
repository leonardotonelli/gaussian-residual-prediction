"""Aggregate verified model summaries under a frozen five-seed analysis contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.five_seed_analysis import analyze_campaign, markdown_report, validate_analysis_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("config/campaigns/five_seed_v1/analysis.yaml"))
    parser.add_argument("--validate-contract", action="store_true", help="Validate a draft contract without loading result files")
    parser.add_argument("--results", type=Path, nargs="+", help="JSON model-result records, or lists of such records")
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are never overwritten")
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text())
    if args.validate_contract:
        if args.results or args.output:
            parser.error("Contract validation does not take results or an output directory")
        validate_analysis_contract(contract, require_frozen=False)
        print(json.dumps({"valid": True, "frozen": contract["frozen"]}))
        return
    if not args.results or args.output is None:
        parser.error("Provide --results and a new --output directory")
    validate_analysis_contract(contract)
    rows = []
    for path in args.results:
        payload = json.loads(path.read_text())
        rows.extend(payload if isinstance(payload, list) else [payload])
    report = analyze_campaign(rows, contract)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output / "summary.md").write_text(markdown_report(report))
    print(json.dumps({"output": str(args.output), "complete": report["complete"], "counts": report["counts"]}))


if __name__ == "__main__":
    main()
