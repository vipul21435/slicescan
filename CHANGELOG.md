# Changelog

## 0.1.0

- Read per-item results and item metadata from JSONL or CSV, joined on the item id.
- Slices from categorical columns, numeric columns in quantile bins and tag columns (overlap
  stated), and two-way intersections with a size-ordered cap.
- `slicescan table`: per-slice score, Wilson or percentile bootstrap interval, Fisher exact or
  permutation test against the rest, Holm and Benjamini-Hochberg adjusted p-values over the
  stated family, beta-binomial shrunk scores and raw-against-shrunk rankings, optional chart.
- `slicescan compare`: paired difference per slice with the Agresti-Min interval and the exact
  McNemar test (sign-flip permutation test for real scores), and sign flips against the overall
  difference.
- `slicescan reversals`: full and majority aggregation reversals over partitioning columns.
- `slicescan explain`: one slice within the table's family.
- Markdown and JSON output.
- Fetch and summary scripts for SWE-bench Verified per-instance results.
- Command-line errors (missing or malformed files, bad scores, unknown models, columns or
  slices, out-of-range options) print one line to stderr and exit with status 2.
- Files with a UTF-8 byte-order mark are read.
- Unknown `--columns` and results whose ids match no metadata row are errors.
- Non-finite scores are rejected with their row number.
- Tied slices share an average rank; Spearman correlation handles ties.
- A majority reversal needs at least two compared slices.
- Exact p-values are exactly 1 when every table counts.
