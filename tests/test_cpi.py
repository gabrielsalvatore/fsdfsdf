from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pie.categories import ALL_ITEMS, CATEGORIES
from pie.cpi import all_items_inflation, load_cpi, price_relatives


def write_cpi(path: Path, rows: list[str]) -> Path:
    path.write_text("category,series_id,date,value\n" + "\n".join(rows) + "\n")
    return path


def test_bundled_file_covers_every_category() -> None:
    cpi = load_cpi()
    assert set(CATEGORIES) | {ALL_ITEMS} <= set(cpi.index)
    assert not cpi.isna().any().any()


def test_bundled_headline_inflation() -> None:
    # 296.311 / 271.696 - 1, the published Jun 2021 -> Jun 2022 CPI-U change
    assert all_items_inflation(load_cpi(), "2021-06", "2022-06") == pytest.approx(0.0906, abs=5e-5)


def test_price_relatives_by_hand(tmp_path: Path) -> None:
    path = write_cpi(
        tmp_path / "cpi.csv",
        ["A,X1,2020-01,100", "A,X1,2021-01,110", "B,X2,2020-01,50", "B,X2,2021-01,45"],
    )
    rel = price_relatives(load_cpi(path), "2020-01", "2021-01", ["B", "A"])
    np.testing.assert_allclose(rel, [0.9, 1.1])


def test_unknown_category_raises(tmp_path: Path) -> None:
    path = write_cpi(tmp_path / "cpi.csv", ["A,X1,2020-01,100", "A,X1,2021-01,110"])
    with pytest.raises(KeyError, match="Z"):
        price_relatives(load_cpi(path), "2020-01", "2021-01", ["A", "Z"])


def test_unknown_date_raises() -> None:
    with pytest.raises(KeyError, match="1999-01"):
        price_relatives(load_cpi(), "1999-01", "2022-06", CATEGORIES)


def test_gap_raises(tmp_path: Path) -> None:
    path = write_cpi(
        tmp_path / "cpi.csv", ["A,X1,2020-01,100", "A,X1,2021-01,110", "B,X2,2020-01,50"]
    )
    with pytest.raises(ValueError, match="gaps"):
        price_relatives(load_cpi(path), "2020-01", "2021-01", ["A", "B"])


def test_duplicate_rows_raise(tmp_path: Path) -> None:
    path = write_cpi(tmp_path / "cpi.csv", ["A,X1,2020-01,100", "A,X1,2020-01,101"])
    with pytest.raises(ValueError, match="duplicate"):
        load_cpi(path)


def test_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "cpi.csv"
    path.write_text("category,date,value\nA,2020-01,100\n")
    with pytest.raises(ValueError, match="series_id"):
        load_cpi(path)
