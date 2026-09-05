# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""One neutral description per benchmark dataset.

Both runners compile their configuration from this file, so the two tools are
given the same columns, the same comparison levels in the same order, the same
blocking, and the same lambda. Anything that differs between the tools after
that is the tools differing, not the benchmark.

Only levels that both libraries implement identically are used -- exact match,
Levenshtein at a distance, Jaro-Winkler at a similarity -- so a pair lands on
the same level in both. Every column is therefore typed as a string; cpplink's
date, numeric and geo levels have no bit-identical splink counterpart and are
out of scope for a parity benchmark.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

# ---------------------------------------------------------------- comparisons


@dataclass(frozen=True)
class Level:
    """A fuzzy level below the exact level. kind is "jw" or "lev"."""

    kind: str
    threshold: float


@dataclass(frozen=True)
class Comparison:
    """null -> exact -> fuzzy levels in order -> else, on a single column."""

    column: str
    levels: tuple = ()
    term_frequency: bool = True

    def to_cpplink(self) -> dict:
        levels = [{"type": "null"}, {"type": "exact"}]
        for level in self.levels:
            name = "jaro_winkler" if level.kind == "jw" else "levenshtein"
            levels.append({"type": name, "threshold": level.threshold})
        levels.append({"type": "else"})
        return {
            "name": self.column,
            "columns": [self.column],
            "term_frequency": self.term_frequency,
            "levels": levels,
        }


# ------------------------------------------------------------------- datasets


@dataclass
class Dataset:
    name: str
    source: str  # attribute on splink_datasets
    rows: int  # expected, for a sanity check in prepare.py
    id_column: str  # in the source frame
    entity: Callable  # frame -> series of true entity ids
    comparisons: List[Comparison]
    block_columns: List[str]  # exact agreement; identical in both tools
    # Conjunctions of columns that are near-certain matches when they all
    # agree. Used once, offline, to derive lambda; see LAMBDA_RECALL.
    lambda_rules: List[tuple] = field(default_factory=list)
    cpplink_extra: List[dict] = field(default_factory=list)
    lam: float = 0.0001  # probability two random records match;
    # refresh with run_splink.py <name> --estimate-lambda
    drop: tuple = ()  # source columns neither tool may see
    rename: Optional[Callable] = None

    @property
    def columns(self) -> List[str]:
        return [c.column for c in self.comparisons]

    def cpplink_schema(self, track: str) -> dict:
        blocking = [{"type": "exact_value", "column": c} for c in self.block_columns]
        if track == "native":
            blocking += self.cpplink_extra
        return {
            "unique_id": "unique_id",
            "columns": [{"name": c, "type": "string"} for c in self.columns],
            "comparisons": [c.to_cpplink() for c in self.comparisons],
            "blocking": blocking,
        }


def _febrl_entity(frame):
    # rec-1496-org and rec-1496-dup-3 are the same person.
    return frame["rec_id"].str.split("-").str[1]


DATASETS = {
    # 1,000 rows. Small enough that the whole pipeline is a smoke test, and
    # every pair could be enumerated, so blocking recall is not the story.
    "fake_1000": Dataset(
        name="fake_1000",
        source="fake_1000",
        rows=1000,
        id_column="unique_id",
        entity=lambda f: f["cluster"].astype(str),
        drop=("cluster",),
        comparisons=[
            Comparison("first_name", (Level("jw", 0.9), Level("jw", 0.8))),
            Comparison("surname", (Level("jw", 0.9), Level("jw", 0.8))),
            Comparison("dob", (Level("lev", 1), Level("lev", 2))),
            Comparison("city", ()),
            Comparison("email", (Level("jw", 0.88),)),
        ],
        block_columns=["dob", "surname", "email"],
        lambda_rules=[("email",), ("first_name", "surname", "dob"), ("surname", "dob")],
        cpplink_extra=[
            {"type": "rare_value", "column": "first_name", "max_frequency": 20},
            {"type": "sorted_neighbourhood", "column": "surname", "window": 10},
        ],
        lam=0.002092092092092092,
    ),
    # 5,000 rows, the FEBRL synthetic deduplication set: one original per
    # entity plus up to five corrupted duplicates, so the corruption model is
    # known and the errors are typographic rather than historical.
    "febrl3": Dataset(
        name="febrl3",
        source="febrl3",
        rows=5000,
        id_column="rec_id",
        entity=_febrl_entity,
        drop=("rec_id",),
        comparisons=[
            Comparison("given_name", (Level("jw", 0.9), Level("jw", 0.8))),
            Comparison("surname", (Level("jw", 0.9), Level("jw", 0.8))),
            Comparison("date_of_birth", (Level("lev", 1), Level("lev", 2))),
            Comparison("soc_sec_id", (Level("lev", 1),)),
            Comparison("address_1", (Level("jw", 0.9), Level("jw", 0.8))),
            Comparison("suburb", (Level("jw", 0.9),)),
            Comparison("postcode", (Level("lev", 1),)),
            Comparison("state", ()),
        ],
        block_columns=["soc_sec_id", "date_of_birth", "surname", "given_name"],
        lambda_rules=[("soc_sec_id",), ("given_name", "surname", "date_of_birth"),
                      ("surname", "date_of_birth", "postcode")],
        cpplink_extra=[
            {"type": "rare_value", "column": "address_1", "max_frequency": 10},
            {"type": "sorted_neighbourhood", "column": "surname", "window": 10},
        ],
        lam=0.0006019203840768153,
    ),
    # 50,578 rows of historical figures with real, messy variation: nicknames,
    # 22% missing dob, 9% missing surname. The main small benchmark.
    "historical_50k": Dataset(
        name="historical_50k",
        source="historical_50k",
        rows=50578,
        id_column="unique_id",
        entity=lambda f: f["cluster"].astype(str),
        drop=("cluster", "full_name"),
        comparisons=[
            Comparison("first_name", (Level("jw", 0.9), Level("jw", 0.8))),
            Comparison("surname", (Level("jw", 0.9), Level("jw", 0.8))),
            Comparison("first_and_surname", (Level("jw", 0.9), Level("jw", 0.8))),
            Comparison("dob", (Level("lev", 1), Level("lev", 2))),
            Comparison("postcode_fake", (Level("lev", 1),)),
            Comparison("birth_place", (Level("jw", 0.9),)),
            Comparison("occupation", ()),
            Comparison("gender", ()),
        ],
        block_columns=["dob", "surname", "postcode_fake", "first_name"],
        lambda_rules=[("first_name", "surname", "dob"),
                      ("first_and_surname", "postcode_fake"),
                      ("surname", "dob", "birth_place")],
        cpplink_extra=[
            {"type": "rare_value", "column": "first_and_surname", "max_frequency": 20},
            {"type": "sorted_neighbourhood", "column": "first_and_surname", "window": 10},
        ],
        lam=7.16903883590421e-05,
    ),
}

# The recall assumed for the deterministic rules when deriving lambda. It is
# the one judgement call in the configuration, it is the same for every dataset
# and both tools get the answer, so it cannot favour either.
LAMBDA_RECALL = 0.8

TRACKS = ("matched", "native")
