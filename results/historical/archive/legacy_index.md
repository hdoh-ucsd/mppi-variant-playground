# Historical recovery index

The current README and `results/variants/` are the primary formulation evidence.
Q01–Q05 retain their showcase figures and seven-state numerical records. Superseded
four/five-state reports and bulky earlier result trees were removed from the current
checkout during repository cleanup; their committed versions remain in Git history.

**Pre-cleanup revision:** `02a331f5e968d230ae6da2c02fda74dd14eed7ba`.

| Historical material | Immutable location |
| --- | --- |
| Earlier mathematical, frozen-decision and trajectory investigations, including reproduction runs | [Original investigation tree](https://github.com/hdoh-ucsd/mppi-variant-playground/tree/02a331f5e968d230ae6da2c02fda74dd14eed7ba/results/investigations) |
| Earlier covariance comparison | [Original comparison tree](https://github.com/hdoh-ucsd/mppi-variant-playground/tree/02a331f5e968d230ae6da2c02fda74dd14eed7ba/results/covariance_comparison) |
| Earlier native and wall-clock outputs | [Native](https://github.com/hdoh-ucsd/mppi-variant-playground/tree/02a331f5e968d230ae6da2c02fda74dd14eed7ba/results/native), [wall-clock](https://github.com/hdoh-ucsd/mppi-variant-playground/tree/02a331f5e968d230ae6da2c02fda74dd14eed7ba/results/wall_clock) |
| Superseded reports and source notes | [Original docs tree](https://github.com/hdoh-ucsd/mppi-variant-playground/tree/02a331f5e968d230ae6da2c02fda74dd14eed7ba/docs) |
| Five-state S-turn primary figure | [Historical figure](https://github.com/hdoh-ucsd/mppi-variant-playground/blob/02a331f5e968d230ae6da2c02fda74dd14eed7ba/results/archive/five_state_Q05_key.png) |
| Earlier run/source-commit inventory | [Original recovery index](https://github.com/hdoh-ucsd/mppi-variant-playground/blob/02a331f5e968d230ae6da2c02fda74dd14eed7ba/results/archive/legacy_index.md) |

To recover committed historical files into a separate directory without repopulating
current `docs/` or `results/`:

```bash
mkdir -p /tmp/mppi-historical-evidence
git archive 02a331f5e968d230ae6da2c02fda74dd14eed7ba docs results \
  | tar -x -C /tmp/mppi-historical-evidence
```

The workspace also retains a **local, Git-ignored** recovery archive at
`runs/archive/02a331f_cleanup/`, including previously untracked pilot data and cached
figures. Its `inventory.json` records moved roots. A fresh clone does not contain
that local archive or other ignored raw executions. Historical internal paths refer
to their original revision; use a checkout of that revision for legacy replay.

Current evidence and reproduction commands are in the [README](../../../README.md),
[consolidated research index](../../../docs/research_index.md), and
[seven-state model report](../../../docs/seven_state_benchmark_report.md).
