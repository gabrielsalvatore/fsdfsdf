# Who Bore the Inflation Surge? — Senior Comp Prototype

Computational artifact for Gabriel Saccoccio's combined Computer Science and Economics senior comprehensive project at Allegheny College (CMPSC 600, Fall 2026).

The prototype calculates a personal inflation rate for every household in the U.S. Bureau of Labor Statistics Consumer Expenditure Survey (CE). It then compares groups of households, with confidence intervals from the survey's replicate weights, and benchmarks three algorithms that compute the same numbers.

## Research question

How did the inflation experienced by U.S. households differ across socioeconomic characteristics, and did those differences change from before COVID-19 to after it?

## What the demo does

1. Loads the CE Interview household files (`fmli*.csv`) for one survey year.
2. Builds each household's spending shares over 16 CPI categories. Owner housing is valued as rental equivalence, as the CPI does.
3. Applies official CPI-U category price changes to get each household's own inflation rate, using the Laspeyres form:

   π_h = Σ_i w_{h,i} · (P_{i,t} / P_{i,0}) − 1

4. Compares groups (urban/rural, income quintile, household type, age, housing tenure). It reports 95% confidence intervals from the 44 BLS replicate weights (balanced repeated replication).
5. Breaks the rural–urban gap into contributions by category.
6. Times three algorithms that produce identical results:

| Algorithm | Idea | Time complexity |
| --- | --- | --- |
| A1 | Naive Python loops | O(H · I · T) |
| A2 | Dense matrix product `W @ R` (vectorized) | O(H · I · T), much lower constant |
| A4 | Aggregate first: `(A @ W) @ R` (matrix-chain reordering) | O(H · I + G · I · T) |

Here H is households, I is items, T is months and G is groups.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt     # to run the demo
pip install -e ".[dev]"             # to develop: adds pytest, hypothesis, ruff, mypy
```

### Get the data

The survey files are public but large (about 60 MB per year), so they are not in this repository.

1. Go to the [CE Public-Use Microdata page](https://www.bls.gov/cex/pumd_data.htm).
2. In the **2023** row, download **Interview** in the **CSV** column (`intrvw23.zip`).
3. Unzip it into this folder, so that `intrvw23/fmli232.csv` exists.

## Run

```bash
python3 -m pie intrvw23                # full demo (python3 pie_demo.py intrvw23 also works)
python3 -m pie intrvw23 --quick        # skip the timing section
python3 -m pie intrvw23 --cpi my.csv   # use a different CPI file
python3 feasibility_check.py intrvw23  # check that every needed variable exists
```

If a CE file lacks a variable the demo needs, it stops with an error naming the
missing variables instead of treating them as zero.

## Develop

```bash
pytest                  # unit tests, including property-based tests of index-number axioms
ruff check . && ruff format --check .
mypy
```

GitHub Actions runs all three on every push and pull request (`.github/workflows/ci.yml`).

## Example results (2023 CE, June 2021 → June 2022)

| Group | Mean household inflation | 95% CI |
| --- | --- | --- |
| Official CPI-U | 9.06% | — |
| Rural | 11.01% | ±0.35 pp |
| Urban | 10.00% | ±0.08 pp |
| Single mothers | 11.04% | ±0.32 pp |
| Single women | 9.22% | ±0.13 pp |
| Lowest income quintile | 10.26% | ±0.16 pp |
| Highest income quintile | 9.24% | ±0.11 pp |

Over December 2019 → December 2023 the rural–urban gap disappears (20.93% vs 21.01%), because gasoline prices fell back.

## Known limitations of the prototype

- Spending shares come from 2023, after the surge. The thesis uses 2016 and 2019 base years. This is the main reason the rebuilt aggregate (9.7%) runs above the official CPI-U (9.1%).
- 16 broad categories built from CE summary variables. The thesis maps detailed UCC codes to a few hundred CPI items using the [BLS CE–CPI concordance](https://www.bls.gov/cpi/additional-resources/ce-cpi-concordance.htm).
- National prices for every household; regional price indexes come later.
- The group comparisons are descriptive. The thesis adds a survey-weighted regression with replicate-weight standard errors.

## Data sources

- U.S. Bureau of Labor Statistics, [Consumer Expenditure Surveys Public-Use Microdata](https://www.bls.gov/cex/pumd_data.htm).
- U.S. Bureau of Labor Statistics, CPI-U category indexes (U.S. city average, not seasonally adjusted), retrieved via [FRED](https://fred.stlouisfed.org/).

## Repository contents

- `pie/`: the prototype as a package.
  - `categories.py`: the 16 CPI categories and the CE variables behind each.
  - `cpi.py`: loads CPI-U category indexes from `data/cpi_u.csv` and computes price relatives.
  - `loading.py`: reads CE FMLI files, checks required variables, builds household spending shares.
  - `weights.py`: weighted quantiles; CPI-style (plutocratic) and democratic aggregate shares.
  - `indexes.py`: household Laspeyres inflation; Laspeyres, Paasche, Fisher and Törnqvist formulas.
  - `groups.py`: assigns households to comparison groups.
  - `aggregation.py`: group averaging and the rural–urban basket-effect decomposition.
  - `variance.py`: BRR standard errors from the 44 replicate weights.
  - `benchmark.py`: times algorithms A1, A2 and A4.
  - `cli.py`: the command-line demo.
- `data/cpi_u.csv`: CPI-U category indexes (category, series ID, month, value).
- `tests/`: pytest suite.
- `pie_demo.py`: backward-compatible entry point that calls `pie.cli`.
- `feasibility_check.py`: confirms the CE files contain every variable the project needs.
- `docs/proposal.md`: the project proposal.
