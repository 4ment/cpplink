# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Fixtures: a small gen-sample file written by the binary, once per session.
Every test needing the binary skips when it is not built."""

import json
import os

import pytest

from cpplink_studio import binary, data

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA = os.path.join(ROOT, "examples", "sample_schema.json")


@pytest.fixture(scope="session")
def cpplink_binary():
    path = binary.find_binary()
    if path is None:
        pytest.skip("cpplink binary not built")
    return path


@pytest.fixture(scope="session")
def sample(tmp_path_factory, cpplink_binary):
    directory = tmp_path_factory.mktemp("sample")
    parquet = str(directory / "sample.parquet")
    truth = str(directory / "sample.truth.csv")
    binary.run(["gen-sample", "--out", parquet, "--rows", "20000", "--truth", truth,
                "--seed", "7"])
    return {"parquet": parquet, "truth": truth}


@pytest.fixture(scope="session")
def sample_schema():
    with open(SCHEMA) as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def dataset(sample):
    return data.Dataset(sample["parquet"])
