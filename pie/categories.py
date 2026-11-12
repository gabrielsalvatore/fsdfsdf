"""CPI categories and the CE summary variables that feed each one."""

from __future__ import annotations

CATEGORIES: list[str] = [
    "Food at home",
    "Food away from home",
    "Alcohol",
    "Shelter",
    "Utilities",
    "Household furnishings",
    "Apparel",
    "Vehicles",
    "Gasoline",
    "Transport services",
    "Medical care",
    "Recreation",
    "Education",
    "Personal care",
    "Tobacco",
    "Other goods/services",
]

ALL_ITEMS = "All items"

# CE summary variable stems per category. Each stem has a current-quarter (CQ)
# and a previous-quarter (PQ) column. Shelter is built separately in
# pie.loading (rent for renters, rental equivalence for owners).
CE_VARS: dict[str, list[str]] = {
    "Food at home": ["FDHOME"],
    "Food away from home": ["FDAWAY"],
    "Alcohol": ["ALCBEV"],
    "Utilities": ["UTIL"],
    "Household furnishings": ["HOUSOP", "HOUSEQ"],
    "Apparel": ["APPAR"],
    "Vehicles": ["CARTKN", "CARTKU", "OTHVEH"],
    "Gasoline": ["GASMO"],
    "Transport services": ["VEHFIN", "MAINRP", "VEHINS", "VRNTLO", "PUBTRA"],
    "Medical care": ["HEALTH"],
    "Recreation": ["ENTERT", "READ"],
    "Education": ["EDUCA"],
    "Personal care": ["PERSCA"],
    "Tobacco": ["TOBACC"],
    "Other goods/services": ["MISC"],
}

QUARTER_SUFFIXES = ("CQ", "PQ")
