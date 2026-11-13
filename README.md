# slicescan

Find the metadata slices where a model's score differs from the rest of an evaluation, with the
multiple-testing correction applied.

Status: early. What works so far is the slice table for one model with 0/1 scores. Paired
comparison of two models, reversal checks, shrinkage, two-way slices, real-valued scores and the
real-data results are not written yet.

## Install

```
pip install git+https://github.com/vipul21435/slicescan
```

## What works so far

- Reading per-item results (`item`, `model`, `score`) and item metadata from JSONL or CSV, joined
  on the item id.
- One-way slices of categorical columns, numeric columns cut into quantile bins, and tag columns
  (an item can carry several tags, so those slices overlap; the output says so).
- For each slice of one model: size, score, Wilson interval, the score of every other item,
  the difference, a two-sided Fisher exact test against the rest, and Holm and
  Benjamini-Hochberg adjusted p-values over every slice tested. Slices under `--min-size` items
  are left out of the family, and the output states the family.
- `slicescan table`, with Markdown or JSON output.

## Quickstart

The example data is synthetic (written by `examples/make_example_data.py`); the system "alpha"
is planted to do worse on the project "kiwi".

```
$ slicescan table examples/data/results.jsonl --meta examples/data/meta.csv --model alpha --columns project
## Slices for `alpha`

Overall: 124/240 = 0.517.

Family: 6 tests (one-way slices of project with at least 10 items); adjusted with Holm (family-wise error) and Benjamini-Hochberg (false discovery rate).

Columns:
- `project`: category, 6 slices

| slice | n | score | 95% CI | rest | diff | p | p Holm | p BH |
|---|---|---|---|---|---|---|---|---|
| project=kiwi | 40 | 0.200 | [0.105, 0.348] | 0.580 | -0.380 | 1.00e-05 | 6.01e-05 | 6.01e-05 |
| project=apple | 40 | 0.650 | [0.495, 0.779] | 0.490 | +0.160 | 0.083 | 0.414 | 0.165 |
| project=lime | 40 | 0.650 | [0.495, 0.779] | 0.490 | +0.160 | 0.083 | 0.414 | 0.165 |
| project=fig | 40 | 0.550 | [0.398, 0.693] | 0.510 | +0.040 | 0.730 | 1.000 | 0.863 |
| project=pear | 40 | 0.550 | [0.398, 0.693] | 0.510 | +0.040 | 0.730 | 1.000 | 0.863 |
| project=plum | 40 | 0.500 | [0.352, 0.648] | 0.520 | -0.020 | 0.863 | 1.000 | 0.863 |

Slices with p < 0.05: 1 before adjustment, 1 after Benjamini-Hochberg, 1 after Holm.
```

## Licence

MIT. See `LICENSE`.

Built by Vipul Raj Jha · https://vipul21435.github.io
