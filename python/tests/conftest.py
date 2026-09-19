# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""One `gen-sample` fixture per session, and the pipeline run over it once.

The fixtures are built with the binding itself, so a failure in `gen_sample`
or `init` shows up here rather than in every test that needs a file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import cpplink

ROWS = 20000
SEED = 7


class Fixture:
    """A sample file, its truth pairs, the drafted schema and a Linker over it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.parquet = root / "sample.parquet"
        self.truth = root / "sample.truth.csv"
        self.schema_path = root / "schema.json"
        cpplink.gen_sample(str(self.parquet), rows=ROWS, truth=str(self.truth))
        self.schema, self.draft = cpplink.init(self.parquet, out=self.schema_path)
        self.linker = cpplink.Linker(self.schema, self.parquet)
        self.model_path = root / "model.json"
        self.model, self.estimate_report = self.linker.estimate(
            out=self.model_path, seed=SEED
        )
        self.predictions = root / "predictions.parquet"
        self.predict_report = self.linker.predict(
            self.model, self.predictions, threshold=10
        )


class LinkFixture:
    """Two files, originals in one and every planted duplicate in the other."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.a = root / "a.parquet"
        self.b = root / "b.parquet"
        self.truth = root / "link.truth.csv"
        cpplink.gen_sample(
            str(self.a), rows=ROWS, truth=str(self.truth), out_b=[str(self.b)]
        )
        self.schema, _ = cpplink.init([self.a, self.b])
        self.linker = cpplink.Linker(self.schema, [self.a, self.b])


@pytest.fixture(scope="session")
def sample(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return Fixture(tmp_path_factory.mktemp("sample"))


@pytest.fixture(scope="session")
def link(tmp_path_factory: pytest.TempPathFactory) -> LinkFixture:
    return LinkFixture(tmp_path_factory.mktemp("link"))
