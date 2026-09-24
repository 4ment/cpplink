# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The cpplink-finder page: who in this file is this person.

    cpplink-finder --schema schema.json --model model.json people.parquet
    python -m streamlit run cpplink_finder/app.py -- people.parquet

A few fields about a person, and the records that score highest against them
under the model, best first. The score is the match weight in bits, which is the
same number `predict` puts on a pair, and the ledger behind every hit is one
click away.

The form is the schema's: a box per column the model compares, in the order the
comparisons are declared, typed by what the column holds. Nothing here computes,
and nothing here is a list of field names -- the weight, the levels and the
term-frequency moves are the core's, because a second implementation of any of
them would be a second answer.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpplink_finder import form as form_spec  # noqa: E402
from cpplink_finder.cli import parser  # noqa: E402
from cpplink_finder.session import Finder, FinderError  # noqa: E402

# Translucent, so they read on a light and a dark theme alike: an opaque pale
# fill would put white text on a light background in the dark one.
STYLES = {
    form_spec.EXACT: "background-color: rgba(46, 160, 67, 0.28)",
    form_spec.PARTIAL: "background-color: rgba(219, 154, 4, 0.30)",
}

st.set_page_config(page_title="cpplink finder", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def load(schema: str, model: str, data: str) -> Finder:
    """One store per process, however many browser sessions there are.

    At the size this page is for, the load is most of a minute and the store is
    gigabytes, so it happens once and every session shares it. That sharing is
    also why the session holds a lock: the searches are serialised.
    """
    return Finder(schema, model, data)


def given() -> dict:
    """The three paths, from the command line where it named them.

    Unknown arguments are left alone rather than refused: the page is a script
    run by streamlit, and what else is on the command line is not its business.
    """
    try:
        args, _ = parser().parse_known_args(sys.argv[1:])
    except SystemExit:
        return {"data": "", "schema": "", "model": ""}
    return {
        "data": args.data or "",
        "schema": args.schema or "",
        "model": args.model or "",
    }


def ask_for_files(paths: dict) -> dict:
    """The three files, in the sidebar, so a page opened bare can still be used."""
    st.sidebar.subheader("Files")
    return {
        "schema": st.sidebar.text_input("Schema", paths["schema"]),
        "model": st.sidebar.text_input("Model", paths["model"]),
        "data": st.sidebar.text_input("Records (parquet)", paths["data"]),
    }


def options(finder: Finder) -> dict:
    """How many hits, how hard to look, and what the score is against.

    These are the settings file's whole contents: what the schema has no opinion
    about. Everything else the page draws comes from the schema itself.
    """
    saved = finder.settings
    st.sidebar.subheader("Search")
    cores = max(1, os.cpu_count() or 1)
    chosen = {
        "k": st.sidebar.slider("Results", 1, 50, int(saved["k"])),
        "threads": st.sidebar.slider(
            "Threads",
            1,
            cores,
            min(cores, int(saved["threads"])),
            help="Splits the pass over the records, which is the floor on a query.",
        ),
        "min_probability": st.sidebar.slider(
            "Minimum probability",
            0.0,
            1.0,
            float(saved["min_probability"]),
            0.01,
            help="Records below this are not returned, however few hits there are.",
        ),
        "expected_matches": st.sidebar.number_input(
            "Expected matches",
            min_value=0.0,
            value=float(saved["expected_matches"]),
            step=1.0,
            help=(
                "How many records here you expect to be this person. It is the "
                "prior: it moves every score by the same amount, never the order."
            ),
        ),
    }
    if st.sidebar.button("Save as default"):
        finder.save_settings(chosen)
        st.sidebar.success(f"Saved beside {os.path.basename(finder.schema_path)}.")
    return chosen


def box(spec: form_spec.Box):
    """One input, as the widget its column's type wants."""
    if spec.kind == "choice":
        return st.selectbox(spec.label, spec.choices, index=0, help=spec.help or None)
    if spec.kind == "date":
        return st.date_input(
            spec.label,
            value=None,
            min_value=pd.Timestamp("1900-01-01").date(),
            max_value=pd.Timestamp.today().date(),
            format="YYYY-MM-DD",
        )
    return st.text_input(spec.label, help=spec.help or None)


def draw_form(finder: Finder) -> tuple[dict, bool]:
    """The boxes, two to a row, in the order the comparisons declare them."""
    values: dict = {}
    with st.form("query"):
        boxes = finder.boxes
        for index in range(0, len(boxes), 2):
            row = st.columns(2)
            for spec, column in zip(boxes[index : index + 2], row, strict=False):
                with column:
                    values[spec.column] = box(spec)
        submitted = st.form_submit_button("Search", type="primary")
    return values, submitted


def table(finder: Finder, outcome) -> pd.DataFrame:
    """The hits as the page shows them: the score, then the record itself."""
    rows = []
    for hit in outcome.hits:
        row = {
            "#": hit.rank,
            "Match weight": round(hit.match_weight, 2),
            "Probability": round(hit.match_probability, 4),
        }
        for column in finder.table_columns:
            row[column] = hit.values.get(column, "")
        rows.append(row)
    return pd.DataFrame(rows)


def draw(finder: Finder, outcome) -> None:
    frame = table(finder, outcome)
    marks = [
        form_spec.cell_classes(finder.form, outcome.query, hit.levels)
        for hit in outcome.hits
    ]
    st.dataframe(
        frame.style.apply(lambda _: form_spec.shading(frame, marks, STYLES), axis=None),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Green: the comparison landed on its exact level. "
        "Amber: it landed on a level between exact and disagreeing."
    )

    report = outcome.report
    st.caption(
        f"{report.records:,} records · {outcome.seconds:.2f} s on "
        f"{report.threads} thread(s) · walk {report.walk_seconds * 1000:,.0f} ms "
        f"over {report.values_walked:,} values · gather "
        f"{report.gather_seconds * 1000:,.0f} ms · {report.rescored:,} scored "
        f"exactly · prior {report.prior:+.1f} bits"
    )

    with st.expander("Why these scores"):
        st.caption(
            "The ledger behind each hit: the prior, then what every comparison "
            "is worth for agreeing or disagreeing, in bits."
        )
        for hit in outcome.hits:
            if hit.waterfall:
                st.text(f"#{hit.rank}  {hit.id}")
                st.code(hit.waterfall, language=None)


def run() -> None:
    st.title("🔎 cpplink finder")
    paths = ask_for_files(given())
    if not all(paths.values()):
        st.info("Give a schema, a model and a parquet file in the sidebar.")
        return
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        st.error(f"no such file: {', '.join(paths[name] for name in missing)}")
        return

    try:
        with st.spinner("Loading the records..."):
            finder = load(paths["schema"], paths["model"], paths["data"])
    except Exception as error:  # noqa: BLE001 - whatever it is, it goes on the page
        st.error(str(error))
        return

    chosen = options(finder)
    st.caption(
        f"{finder.records:,} records from {os.path.basename(finder.data_path)} · "
        f"{len(finder.boxes)} fields the model compares"
    )

    values, submitted = draw_form(finder)
    if submitted:
        try:
            with st.spinner("Searching..."):
                st.session_state.outcome = finder.search(values, **chosen)
        except (FinderError, form_spec.FormError) as error:
            st.warning(str(error))
            st.session_state.pop("outcome", None)

    outcome = st.session_state.get("outcome")
    if outcome is None:
        return
    if not outcome.hits:
        st.warning("No record here scores above the minimum.")
        return
    draw(finder, outcome)


run()
