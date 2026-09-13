# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""cpplink-studio: explore a file, draft its schema, and price its blocking.

The package is the engine; ``app.py`` is a thin Streamlit page over it and
``cli.py`` drives the same functions headlessly. Nothing here imports the UI,
so every number the page shows can be reproduced and tested without it.
"""

__version__ = "0.1.0"
