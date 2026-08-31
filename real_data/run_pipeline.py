#!/usr/bin/env python3
"""
The actual test: feed real ES features through generator -> red_team ->
risk_arbiter, and report exactly what survives.

Uses the DEFAULT heuristic scorer (rank-IC based), not the learned scorer -
core.scorer.LearnedScorer is trained on a synthetic corpus and fitting it on
this small a real sample would just be a second layer of "grading its own
homework". That's future work (ASSESSMENT.md item #2), not this pass.
"""
import json
from pathlib import Path

if __package__ in (None, ""):
    import pathlib as _pl, sys as _sys
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "stark-core"))

from core.generator import Generator
from core.red_team import RedTeam
from core.risk_arbiter import Proposal, RiskArbiter

FEATURES_FILE = Path(__file__).parent / "es_features.json"


def main():
    data = json.loads(FEATURES_FILE.read_text())
    features = data["features"]
    forward_returns = data["forward_returns"]
    n = data["n_bars"]

    print("STARK MK18 - REAL DATA RUN")
    print(f"Source: {data['n_raw_ticks']} real Rithmic ES ticks, "
          f"{data['ticks_per_bar']} ticks/bar, {n} bars.")
    print(f"Features: {list(features.keys())}\n")

    gen = Generator(seed=1, min_prior=0.0)  # min_prior=0.0: show everything, let red team decide
    hypotheses = gen.generate(features, forward_returns, horizon=1)
    print(f"Generator considered {len(features)} features, produced {len(hypotheses)} hypotheses.\n")

    team = RedTeam(cost_bps=5.0, seed=42)
    arbiter = RiskArbiter()

    survivors = []
    for h in hypotheses:
        print(f"=== {h} ===")
        report = team.run(h, features[h.feature], forward_returns)
        for a in report.attacks:
            print(a)

        adjusted = report.adjusted_confidence(h.prior)
        state = "SURVIVED" if report.survived else f"KILLED by {', '.join(a.name for a in report.kill_shots)}"
        print(f"  => {state} | confidence {h.prior:.3f} -> {adjusted:.3f}")

        proposal = Proposal(
            hypothesis_id=h.id,
            size_fraction=0.05,
            confidence=adjusted,
            expected_edge_bps=h.edge_bps - team.cost_bps,  # net of the red team's own cost assumption
            n_trades=h.n_obs,
            survived_red_team=report.survived,
            volatility_bps=h.vol_bps,  # let the arbiter's own unmeasurable-variance rule catch 0/non-finite
        )
        ruling = arbiter.rule(proposal)
        print(f"  arbiter: {str(ruling).replace(chr(10), chr(10) + '    ')}")
        print()

        if report.survived and ruling.decision != "VETO":
            survivors.append((h, report, ruling))

    print("=" * 78)
    print(f"RESULT: {len(survivors)} of {len(hypotheses)} hypotheses survived generator -> "
          f"red team -> arbiter.")
    if not survivors:
        print("Nothing survived. On real data, that is not a bug - it is the honest, "
              "expected result for a system whose every prior validation was against "
              "its own synthetic generators.")
    for h, report, ruling in survivors:
        print(f"  SURVIVOR: {h.feature} | confidence {report.adjusted_confidence(h.prior):.3f} "
              f"| arbiter: {ruling.decision} @ {ruling.max_size_fraction:.4f}")


if __name__ == "__main__":
    main()
