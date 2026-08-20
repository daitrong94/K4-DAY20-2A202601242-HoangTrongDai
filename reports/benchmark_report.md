# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline: Research GraphRAG state-of-the-art an... | 10.70 | 0.0004 | 6.0 |  | 0% |  |
| multi-agent: Research GraphRAG state-of-the-art an... | 24.02 | 0.0015 | 10.0 | 100% | 0% |  |
| baseline: Compare single-agent and multi-agent ... | 8.89 | 0.0002 | 6.0 |  | 0% |  |
| multi-agent: Compare single-agent and multi-agent ... | 24.38 | 0.0015 | 10.0 | 100% | 0% |  |
| baseline: Summarize production guardrails for L... | 4.20 | 0.0002 | 6.0 |  | 0% |  |
| multi-agent: Summarize production guardrails for L... | 30.60 | 0.0015 | 10.0 | 100% | 0% |  |

## Summary

- Runs: 6
- Avg latency: 17.13s
- Total estimated cost: $0.0052
- Avg quality (automated proxy): 8.0/10
- Failure rate: 0%

> Quality above is a cheap automated proxy (see `evaluation/benchmark.py`). Pair it with the human rubric in `docs/peer_review_rubric.md`, and attach a trace link or screenshot per run when submitting this report.
