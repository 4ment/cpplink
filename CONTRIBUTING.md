# Contributing

This page is the developer's side of the project: how to build with the tests on, what the checks are, and how the documentation is built.
Installing and running the tool is in the [README](README.md) and the [documentation](https://4ment.github.io/cpplink/).

## Environment

Everything but the compiler comes from the project's conda environment, declared in [environment.yml](environment.yml): CMake, Arrow C++, `nlohmann_json`, GoogleTest, the lint tools and MkDocs.
Builds use the system `clang++`.

```sh
conda env create -f environment.yml
conda activate cpplink
```

## Building with the tests

```sh
cmake -S . -B build -DBUILD_TESTING=on -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build build
ctest --test-dir build                          # the whole suite
ctest --test-dir build -R RejectsUnknown -V     # one test, by regex on its gtest name
./build/run_tests --gtest_filter='RunTest.*'    # or drive the gtest binary directly
```

`-DCMAKE_PREFIX_PATH=$CONDA_PREFIX` is what makes `find_package(GTest)` succeed; without it the test configure step fails even inside the activated environment.

Tests live in `tests/*_test.cpp`, are linked against the `cpplink_core` static library, and drive `cpplink::Run(args, out, err)` with string streams where they exercise a command.
Sources are globbed with `CONFIGURE_DEPENDS`, so a new `src/cpplink/*.cpp` or `tests/*_test.cpp` needs no `CMakeLists.txt` edit, only a rebuild.

### Presets and sanitizers

[CMakePresets.json](CMakePresets.json) holds the same configuration as presets, with tests on: `release` builds into `build/`, the others into `build-<preset>/`.
Two of them run the suite under a sanitizer, which is what checks the claim that the threaded stages share the record store with no lock rather than asserting it:

```sh
cmake --preset asan && cmake --build --preset asan && ctest --preset asan   # ASan + UBSan, about 30 s
cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan   # ThreadSanitizer, about 5 min
```

Both are clean over the whole suite.
CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs `release`, `asan` and `tsan` on Linux and macOS, then the Python suite and the format and lint checks below.

## The Python package

```sh
pip install -e . --no-build-isolation -Ccmake.define.CMAKE_PREFIX_PATH=$CONDA_PREFIX
python -m pytest python/tests
```

The editable install redirects the Python sources to `python/cpplink/`, so a change there needs no rebuild; a change to `python/bindings/` or to the core needs the `pip install` line again.
The binding is a second front end over the same stage functions, not a second implementation, and `python/tests/test_parity.py` holds it to the command line's output byte for byte.

## Code style

C++ follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with two customizations, a 4-space indent and a 90-column limit, both in [.clang-format](.clang-format) and [CPPLINT.cfg](CPPLINT.cfg).
Every source and header opens with the two-line copyright and SPDX banner, which is also what keeps cpplint's `legal/copyright` check quiet.

```sh
cmake --build build --target format                                                   # clang-format, in place
clang-format --dry-run --Werror src/main.cpp src/cpplink/* tests/*.cpp python/bindings/*   # check only
cpplint --recursive src tests python/bindings                                         # zero errors, keep it so
ruff check .                                                                          # the Python code
ruff format .                                                                         # or --check
```

Two cpplint rules are switched off in `CPPLINT.cfg` on purpose: `build/c++11`, because the project is C++17, and `build/include_order`, because cpplint reads any `<foo.h>` as a C system header and third-party headers could never satisfy it without contradicting Google style.

## Documentation

The site is built with [MkDocs](https://www.mkdocs.org/) and Material from [docs/](docs/); the navigation is in [mkdocs.yml](mkdocs.yml).

```sh
mkdocs serve             # live preview on http://127.0.0.1:8000
mkdocs build --strict    # render to site/, failing on a broken link or anchor
```

Every measurement quoted in the docs was produced by the commands documented there, and a page that states a number says which dataset and how many threads it came from.
Keep it that way: a claim about behaviour belongs beside the command that demonstrates it.

Two conventions for the markdown: no em dashes, and one sentence per line.
