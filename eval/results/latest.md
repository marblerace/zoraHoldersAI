# SQL evaluation status

The golden set now contains 44 cases: 37 executable SQL questions and 7 clarification cases.

The first expanded run on 2026-07-18 was excluded from published quality metrics because the
Claude Code subscription hit its session limit at case q027. The remaining near-zero-latency
failures were circuit-breaker fallbacks, not SQL-generation results.

Run the following after the subscription window resets to regenerate this report and the README
table from a complete measurement:

```bash
.venv/bin/python -m eval.run
```

The last complete judge calibration measured answer groundedness at 86.84%, up from 26.32% before
the faithfulness/scoring fixes. The expanded set adds three multi-relation SQL cases and three
unsupported-or-ambiguous clarification cases.
