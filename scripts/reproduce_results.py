"""Verify the preserved exports and reproduce the paper's primary analysis offline."""
import argparse
import hashlib
import json
from pathlib import Path

from iwm_replication.five_seed_analysis import analyze_campaign, markdown_report

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/campaigns/five_seed_v1/20260925_eval_fix_v3"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/reproduced-analysis")
    args = p.parse_args()
    provenance = json.loads((EVIDENCE / "provenance.json").read_text())
    matched, omitted = [], []
    for record in provenance["exports"]:
        name = Path(record["path"]).name
        path = EVIDENCE / name
        if Path(name).suffix not in (".csv", ".json"):
            omitted.append(name)  # Earlier plot exports and cluster-only notes are not release inputs.
            continue
        if name == "training_curves.csv" and not path.exists():
            continue  # Explicitly unavailable in the preprint.
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Export checksum mismatch: " + name)
        matched.append(name)
    report = analyze_campaign(json.loads((EVIDENCE / "model_results.json").read_text()),
                              json.loads((EVIDENCE / "frozen_analysis.json").read_text()))
    original = json.loads((EVIDENCE / "analysis.json").read_text())
    def compare(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a: compare(a[k], b[k])
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b): compare(x, y)
        elif isinstance(a, float):
            import math
            assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
        else: assert a == b
    compare(report, original)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "summary.md").write_text(markdown_report(report))
    print(json.dumps({"matched_export_hashes": len(matched), "omitted_auxiliary_exports": omitted, "counts": report["counts"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
