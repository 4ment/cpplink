# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""The cpplink-studio page: explore a file, draft its schema, price its blocking.

    cpplink-studio data.parquet
    python -m streamlit run cpplink_studio/app.py -- data.parquet

Six tabs. **Data** opens a csv or parquet through DuckDB, shows the types it
read and writes the typed parquet the binary needs. **Columns** is the grid of
type and role dropdowns over the column statistics, with a detail view per
column. **Correlation** is what pairs of columns share, on a sample, and the
profile ledger from the binary. **Comparisons** edits the levels the roles
proposed. **Blocking** edits the plan, prices every source live from the value
counts, and draws what a knob would do. **Schema** shows, saves and validates
the JSON.

Nothing here computes: the page reads the engine modules and keeps one draft
in ``st.session_state``. Numbers only the binary can produce come from the
binary, through ``binary.py``, and are labelled as such.
"""

import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpplink_studio import binary, blocking, data, guess, schema, stats, templates  # noqa: E402
from cpplink_studio.cli import analyse, guess_unique_id  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9892"
LEVEL_TYPES = ["null", "exact", "levenshtein", "jaro_winkler", "date_within",
               "numeric_within", "geo_within", "list_overlap", "list_jaccard",
               "list_contains", "list_levenshtein", "list_jaro_winkler",
               "contains_levenshtein", "contains_jaro_winkler", "else"]
TRANSFORMS = ["normalize", "sorted_tokens", "soundex", "year", "month", "day",
              "year_month", "email_username", "email_domain"]
BUDGETS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

st.set_page_config(page_title="cpplink studio", page_icon="🔗", layout="wide")


# --- state --------------------------------------------------------------------

def state():
    if "studio" not in st.session_state:
        st.session_state.studio = {"editor_version": 0, "pairs": {}, "reports": {}}
    return st.session_state.studio


def bump():
    """Editors are keyed by this counter, so a programmatic change to the draft
    replaces what they show instead of being overlaid by their stale edits."""
    state()["editor_version"] += 1


def ekey(name):
    return f"{name}-{state()['editor_version']}"


def load_dataset(path, all_text=False):
    s = state()
    options = {"all_varchar": "true"} if all_text else None
    with st.spinner(f"Reading {os.path.basename(path)}"):
        ds = data.Dataset(path, options)
        pricer = blocking.Pricer(ds, {})
        types, column_stats, probes, roles, reasons = analyse(ds, pricer)
    s.update({"ds": ds, "stats": column_stats, "probes": probes, "reasons": reasons,
              "pairs": {}, "reports": {}, "typed_path": None})
    unique_id = guess_unique_id(ds, roles, column_stats)
    draft = schema.new_draft(ds, types, roles, unique_id)
    schema.apply_templates(draft)
    s["draft"] = draft
    pricer.types = declared_types(draft)
    s["pricer"] = pricer
    redraft_plan(BUDGETS[5])
    bump()


def redraft_plan(budget_per_row):
    s = state()
    draft = s["draft"]
    included = {c["name"]: c["type"] for c in draft["columns"] if c["include"]}
    roles = {c["name"]: c["role"] for c in draft["columns"] if c["include"]}
    plan, reasons = blocking.draft_plan(s["pricer"], roles, included,
                                        schema.to_schema(draft)["columns"], s["ds"].rows,
                                        budget_per_row, draft["derived"])
    draft["blocking"] = plan
    s["plan_reasons"] = reasons


def declared_names(draft):
    names = [c["name"] for c in draft["columns"] if c["include"]]
    return names + [d["name"] for d in draft["derived"]]


def declared_types(draft):
    """Every transform produces a string, so a derived column is one."""
    types = {c["name"]: c["type"] for c in draft["columns"] if c["include"]}
    for d in draft["derived"]:
        types[d["name"]] = "string"
    return types


def data_paths_for_binary():
    """The parquet the binary reads: the input when it can, else the typed copy
    the Data tab wrote. None with a message when neither will do."""
    s = state()
    ds, draft = s["ds"], s["draft"]
    needs = [c["name"] for c in draft["columns"] if c["include"] and
             ds.needs_cast(c["name"], c["type"])]
    if draft.get("unique_id") and ds.needs_cast(draft["unique_id"], "string"):
        needs.append(draft["unique_id"])
    if ds.kind == "parquet" and not needs:
        return [ds.path], None
    if s.get("typed_path") and os.path.exists(s["typed_path"]):
        return [s["typed_path"]], None
    why = "the file is csv" if ds.kind == "csv" else \
        f"{', '.join(needs)} need a cast"
    return None, f"cpplink cannot read this file directly ({why}): write the typed " \
                 f"parquet from the Data tab first"


def fmt_int(n):
    return f"{int(n):,}"


# --- charts -------------------------------------------------------------------

def bar_figure(labels, values, title, color=BLUE, log=False, line=None, unit=""):
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=color,
                           marker_line_width=0, hovertemplate="%{y}: %{x:,}" + unit +
                           "<extra></extra>"))
    fig.update_layout(title=title, height=max(220, 34 * len(labels) + 90),
                      margin=dict(l=10, r=10, t=40, b=10), bargap=0.35,
                      yaxis=dict(autorange="reversed"),
                      xaxis=dict(type="log" if log else "linear", showgrid=True,
                                 gridcolor="rgba(128,128,128,0.2)"))
    if line is not None:
        fig.add_vline(x=line, line_dash="dot", line_color=ORANGE,
                      annotation_text="budget", annotation_position="top")
    return fig


def curve_figure(x, y, title, xlabel, ylabel, mark=None, logx=True, logy=True,
                 color=BLUE, hover=None):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=color,
                               width=2), marker=dict(size=8),
                               hovertemplate=hover or "%{x}: %{y:,}<extra></extra>"))
    if mark is not None:
        fig.add_vline(x=mark, line_dash="dot", line_color=ORANGE,
                      annotation_text="current", annotation_position="top")
    fig.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(title=xlabel, type="log" if logx else "linear"),
                      yaxis=dict(title=ylabel, type="log" if logy else "linear",
                                 gridcolor="rgba(128,128,128,0.2)"))
    return fig


# --- sidebar ------------------------------------------------------------------

def sidebar():
    s = state()
    st.sidebar.title("cpplink studio")
    default = sys.argv[1] if len(sys.argv) > 1 else ""
    path = st.sidebar.text_input("Data file (csv or parquet)", value=default)
    all_text = st.sidebar.checkbox("Read every csv column as text", value=False,
                                   help="Skip DuckDB's type sniffing; pick every type "
                                        "yourself in the Columns tab.")
    if st.sidebar.button("Load", type="primary", width="stretch"):
        if not path or not os.path.exists(path):
            st.sidebar.error("no such file")
        else:
            try:
                load_dataset(path, all_text)
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(str(exc))
    if "ds" in s:
        ds = s["ds"]
        st.sidebar.caption(f"{fmt_int(ds.rows)} rows, {len(ds.columns)} columns, "
                           f"{ds.kind}")
    st.sidebar.divider()
    existing = st.sidebar.text_input("Open an existing schema", value="")
    if st.sidebar.button("Open schema", width="stretch", disabled="ds" not in s):
        try:
            with open(existing) as handle:
                loaded = json.load(handle)
            draft = schema.from_schema(loaded, s["ds"])
            for column in draft["columns"]:
                column["role"] = s["draft"]["columns"][
                    [c["name"] for c in s["draft"]["columns"]].index(column["name"])
                ]["role"] if column["name"] in s["ds"].columns else "none"
            s["draft"] = draft
            s["pricer"].types = declared_types(draft)
            bump()
            st.sidebar.success("schema opened")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(str(exc))
    s["truth"] = st.sidebar.text_input("Truth pairs (csv, optional)", value="")
    s["threads"] = st.sidebar.number_input("Threads for the wall estimate", 1, 256,
                                           max(1, os.cpu_count() or 1))
    st.sidebar.divider()
    found = binary.find_binary()
    if found:
        st.sidebar.caption(f"binary: `{found}`")
    else:
        st.sidebar.warning("cpplink binary not found: set $CPPLINK or build it")


# --- Data tab -----------------------------------------------------------------

def data_tab():
    s = state()
    ds, draft = s["ds"], s["draft"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", fmt_int(ds.rows))
    c2.metric("Columns", len(ds.columns))
    size = data.file_size(ds.path)
    c3.metric("File", f"{size / 1e6:,.1f} MB" if size else "?")
    c4.metric("Pairs unblocked", f"{blocking.pairs_in(ds.rows):.3g}")
    st.dataframe(ds.head(20), width="stretch", height=320)

    st.subheader("Types")
    rows = []
    for c in draft["columns"]:
        arrow = ds.arrow_types.get(c["name"])
        rows.append({"column": c["name"], "file type": ds.duckdb_types[c["name"]],
                     "arrow type": str(arrow) if arrow is not None else "",
                     "cpplink type": c["type"],
                     "needs cast": ds.needs_cast(c["name"], c["type"])})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("cpplink reads parquet only, and its loader wants `string`, `date32`/"
               "`date64`/`timestamp`, `double`/`float`, `bool` and `list<string>` "
               "exactly, with a string `unique_id`. A column marked as needing a "
               "cast is rewritten by the export below.")

    st.subheader("Write the typed parquet")
    st.write("The file cpplink will read: every included column cast to its type, "
             "blanks and the sentinels you pick as missing, the id as a string.")
    out = st.text_input("Output path", value=data.default_output_path(ds.path))
    with st.expander("Cast options per column"):
        for c in draft["columns"]:
            if not c["include"]:
                continue
            opts = draft["cast_options"].setdefault(c["name"], {})
            cols = st.columns([2, 3, 2, 2])
            cols[0].write(f"**{c['name']}** ({c['type']})")
            sentinels = cols[1].text_input("Values read as missing (comma separated)",
                                           value=", ".join(opts.get("sentinels", [])),
                                           key=f"sent-{c['name']}")
            opts["sentinels"] = [v.strip() for v in sentinels.split(",") if v.strip()]
            if c["type"] == "date" and ds.duckdb_types[c["name"]].upper() in (
                    "VARCHAR", "TEXT", "STRING"):
                opts["date_format"] = cols[2].text_input(
                    "strptime format", value=opts.get("date_format", ""),
                    key=f"fmt-{c['name']}", help="Empty means ISO") or None
            if c["type"] == "string_list" and not stats.is_list_type(
                    ds.duckdb_types[c["name"]]):
                opts["list_separator"] = cols[3].text_input(
                    "List separator", value=opts.get("list_separator", "|"),
                    key=f"sep-{c['name']}")
    if st.button("Check cast failures"):
        rows = []
        for c in draft["columns"]:
            if c["include"]:
                failures = data.cast_failures(ds, c["name"], c["type"],
                                              draft["cast_options"].get(c["name"]))
                rows.append({"column": c["name"], "type": c["type"],
                             "values lost to the cast": failures})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    if st.button("Write typed parquet", type="primary"):
        types = {c["name"]: c["type"] for c in draft["columns"]}
        include = [c["name"] for c in draft["columns"] if c["include"]]
        with st.spinner("writing"):
            id_name = data.write_typed_parquet(ds, out, types, draft.get("unique_id"),
                                               draft["cast_options"], include)
        draft["unique_id"] = id_name
        s["typed_path"] = out
        st.success(f"wrote {out}; the binary now reads this file")
    if s.get("typed_path"):
        st.caption(f"typed parquet: `{s['typed_path']}`")


# --- Columns tab --------------------------------------------------------------

def columns_tab():
    s = state()
    ds, draft, column_stats = s["ds"], s["draft"], s["stats"]
    ids = [None] + list(ds.columns)
    current = draft.get("unique_id")
    picked = st.selectbox("unique_id column", ids,
                          index=ids.index(current) if current in ids else 0,
                          help="Must be unique and non-null; written as a string. "
                               "None writes a row number.")
    if picked != current:
        draft["unique_id"] = picked
        for c in draft["columns"]:
            if c["name"] == picked:
                c["include"] = False
        bump()
        st.rerun()

    rows = []
    for c in draft["columns"]:
        st_ = column_stats.get(c["name"], {})
        confidence, reason = s["reasons"].get(c["name"], ("", ""))
        top = st_.get("overrepresented") or []
        rows.append({
            "include": c["include"], "column": c["name"], "type": c["type"],
            "role": c["role"], "confidence": confidence, "why": reason,
            "file type": ds.duckdb_types.get(c["name"], ""),
            "null %": 100 * st_.get("null_share", 0.0),
            "distinct": st_.get("distinct"),
            "effective": st_.get("effective_cardinality"),
            "bits": st_.get("bits"),
            "top value share %": 100 * st_.get("top_share", 0.0),
            "overrepresented": len(top), "sentinels": len(st_.get("sentinels") or []),
        })
    edited = st.data_editor(
        pd.DataFrame(rows), key=ekey("columns"), hide_index=True, width="stretch",
        height=35 * (len(rows) + 1) + 3,
        column_config={
            "include": st.column_config.CheckboxColumn("include"),
            "type": st.column_config.SelectboxColumn("type", options=data.CPPLINK_TYPES,
                                                     required=True),
            "role": st.column_config.SelectboxColumn("role", options=guess.ROLES,
                                                     required=True),
            "null %": st.column_config.NumberColumn(format="%.2f"),
            "distinct": st.column_config.NumberColumn(format="localized"),
            "effective": st.column_config.NumberColumn(
                "effective cardinality", format="localized",
                help="1 / P(two random rows agree): what the column is worth as a "
                     "number of equally likely values"),
            "bits": st.column_config.NumberColumn(
                format="%.2f", help="-log2 of the collision rate: the most an exact "
                                    "agreement on this column can be worth"),
            "top value share %": st.column_config.NumberColumn(format="%.2f"),
        },
        disabled=["column", "confidence", "why", "file type", "null %", "distinct",
                  "effective", "bits", "top value share %", "overrepresented",
                  "sentinels"])
    changed = False
    for c, (_, row) in zip(draft["columns"], edited.iterrows()):
        for key in ("include", "type", "role"):
            if c[key] != row[key]:
                c[key] = row[key]
                changed = True
    if changed:
        s["pricer"].types = declared_types(draft)

    b1, b2, _ = st.columns([2, 3, 5])
    if b1.button("Guess roles again"):
        types = {c["name"]: c["type"] for c in draft["columns"]}
        for c in draft["columns"]:
            role, confidence, reason = guess.guess_role(
                c["name"], s["probes"][c["name"]], column_stats[c["name"]], types[c["name"]])
            c["role"] = role
            s["reasons"][c["name"]] = (confidence, reason)
        bump()
        st.rerun()
    if b2.button("Rebuild comparisons and blocking from the roles"):
        schema.apply_templates(draft)
        s["pricer"].types = declared_types(draft)
        redraft_plan(s.get("budget", BUDGETS[5]))
        bump()
        st.rerun()

    st.subheader("Column detail")
    included = [c["name"] for c in draft["columns"] if c["include"]] or list(ds.columns)
    column = st.selectbox("Column", ds.columns, key="detail-column",
                          index=ds.columns.index(included[0]))
    st_ = column_stats[column]
    left, right = st.columns([3, 2])
    top = stats.top_values(ds, column, 20)
    non_null = st_["rows"] - st_["nulls"]
    if top:
        fig = bar_figure([str(v) for v, _ in top], [c for _, c in top],
                         f"Top values of {column}", unit=" rows")
        left.plotly_chart(fig, width="stretch")
    detail = {k: (f"{v:,.4g}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int)
                  else str(v))
              for k, v in st_.items() if k not in ("overrepresented", "sentinels", "column")}
    right.dataframe(pd.Series(detail, name=column).to_frame(), height=35 * (len(detail) + 1))
    if st_["overrepresented"]:
        st.warning("Overrepresented values, holding far more rows than a value of this "
                   "column typically does: " + ", ".join(
                       f"`{v}` ({100 * share:.2f}%)" for v, _, share in
                       st_["overrepresented"]))
    if st_["sentinels"]:
        st.info("Values that usually mean missing: " + ", ".join(
            f"`{v}` ({fmt_int(c)})" for v, c in st_["sentinels"]) +
            ". Add them under cast options in the Data tab to write them as null.")
    counts, _ = s["pricer"].counts(column)
    hist = blocking.group_size_histogram(counts)
    if hist and blocking.exact_pairs(counts) > 0:
        fig = go.Figure(go.Bar(x=[f"{b['low']}–{b['high']}" for b in hist],
                               y=[b["pairs"] for b in hist], marker_color=BLUE,
                               customdata=[[b["values"], b["rows"]] for b in hist],
                               hovertemplate="group size %{x}<br>%{customdata[0]:,} "
                                             "values, %{customdata[1]:,} rows<br>"
                                             "%{y:,} pairs<extra></extra>"))
        fig.update_layout(title=f"Pairs by group size of {column}", height=300,
                          margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title="rows sharing the value",
                          yaxis=dict(title="pairs from those groups", type="log"))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{fmt_int(non_null)} non-null rows in {fmt_int(len(counts))} groups; "
                   f"exact agreement would enumerate "
                   f"{fmt_int(blocking.exact_pairs(counts))} pairs.")


# --- Correlation tab ----------------------------------------------------------

def correlation_tab():
    s = state()
    ds, draft = s["ds"], s["draft"]
    discrete = [c["name"] for c in draft["columns"] if c["include"] and
                c["type"] != "double" and not stats.is_list_type(
                    ds.duckdb_types[c["name"]])]
    chosen = st.multiselect("Columns", discrete, default=discrete[:12])
    st.caption("Computed on a sample of up to 1,000,000 rows. Containment: one "
               "column's text inside the other's. Determination: how far one column's "
               "value fixes the other's, against the share its commonest value has "
               "anyway. Overlap: bits by which agreeing on both is more likely than "
               "independence would make it, the evidence a model would count twice; "
               "unreliable where the file's own duplicates are the only collisions.")
    if st.button("Compute", type="primary") and len(chosen) >= 2:
        with st.spinner("sampling"):
            n = stats.sample_table(ds)
            s["pair_sample_rows"] = n
            for i, a in enumerate(chosen):
                for b in chosen[i + 1:]:
                    if (a, b) not in s["pairs"]:
                        s["pairs"][(a, b)] = stats.pair_stats(ds, a, b)
    pairs = {k: v for k, v in s["pairs"].items() if k[0] in chosen and k[1] in chosen
             and v is not None}
    if pairs:
        floor = stats.flag_reliability(pairs)
        st.caption(f"sample: {fmt_int(s.get('pair_sample_rows', 0))} rows. At least "
                   f"{fmt_int(floor)} pairs of rows agree on two columns beyond what "
                   f"chance allows: the file's duplicates, which put a floor under every "
                   f"joint. An overlap is trusted only where independence alone would "
                   f"produce five times that many collisions.")
        names = chosen
        z = [[None] * len(names) for _ in names]
        text = [[""] * len(names) for _ in names]
        for (a, b), p in pairs.items():
            i, j = names.index(a), names.index(b)
            bits_ = p.get("overlap_bits")
            if bits_ is not None and p.get("overlap_reliable"):
                z[i][j] = z[j][i] = max(bits_, 0.0)
                text[i][j] = text[j][i] = f"{bits_:.2f}"
            elif bits_ is not None:
                text[i][j] = text[j][i] = "dup"
        fig = go.Figure(go.Heatmap(z=z, x=names, y=names, text=text,
                                   texttemplate="%{text}", colorscale="Blues",
                                   hoverongaps=False, zmin=0,
                                   hovertemplate="%{y} × %{x}: %{text} bits<extra></extra>"))
        fig.update_layout(title="U-side overlap, bits (blank: independent or unmeasurable; "
                                "dup: the duplicates swamp the joint)",
                          height=120 + 40 * len(names), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        rows = []
        for (a, b), p in pairs.items():
            rows.append({
                "a": a, "b": b,
                "a in b %": 100 * (p.get("containment_a_in_b") or 0),
                "b in a %": 100 * (p.get("containment_b_in_a") or 0),
                "a→b determination": p.get("determination_a_to_b"),
                "b baseline": p.get("baseline_b"),
                "b→a determination": p.get("determination_b_to_a"),
                "a baseline": p.get("baseline_a"),
                "overlap bits": p.get("overlap_bits"),
                "reliable": p.get("overlap_reliable"),
                "joint collisions": p.get("observed_joint_collisions"),
                "expected if independent": p.get("expected_joint_collisions"),
                "b null | a null": p.get("null_b_given_null_a"),
                "b null": p.get("null_b"),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={k: st.column_config.NumberColumn(format="%.3f")
                                    for k in ("a→b determination", "b baseline",
                                              "b→a determination", "a baseline",
                                              "overlap bits", "b null | a null", "b null",
                                              "joint collisions",
                                              "expected if independent")} | {
                         "a in b %": st.column_config.NumberColumn(format="%.1f"),
                         "b in a %": st.column_config.NumberColumn(format="%.1f")})
        flagged = [(a, b) for (a, b), p in pairs.items() if
                   max(p.get("containment_a_in_b") or 0, p.get("containment_b_in_a") or 0)
                   >= 0.9]
        if flagged:
            st.warning("Containment on 90% of rows, which `estimate` treats as a tie "
                       "and holds out together: " +
                       ", ".join(f"{a} / {b}" for a, b in flagged))

    st.subheader("cpplink profile")
    st.write("The evidence ledger and the dependence map from the binary, over the "
             "whole file and the schema as drafted: what each column can be worth, "
             "what a matching pair would score, and which pairs of columns are the "
             "same evidence twice.")
    profile_button("profile")


def profile_button(key):
    s = state()
    paths, why = data_paths_for_binary()
    if why:
        st.info(why)
        return
    if st.button("Run cpplink profile", key=key):
        try:
            with st.spinner("profiling"):
                report, err, elapsed = binary.profile(schema.to_json(s["draft"]), paths)
            s["reports"]["profile"] = (report, elapsed)
        except binary.BinaryError as exc:
            st.error(str(exc))
    if "profile" in s["reports"]:
        report, elapsed = s["reports"]["profile"]
        st.caption(f"{elapsed:.1f} s")
        for key_, value in report.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                st.write(f"**{key_}**")
                st.dataframe(pd.DataFrame(value), width="stretch", hide_index=True)
        with st.expander("raw report"):
            st.json(report)


# --- Comparisons tab ----------------------------------------------------------

def comparisons_tab():
    s = state()
    draft = s["draft"]
    names = declared_names(draft)
    types = declared_types(draft)

    st.subheader("Derived columns")
    st.caption("Declared, not written: a transform runs once per distinct value at "
               "load, and the result is interned, counted, blocked on and compared "
               "like any column.")
    derived_df = pd.DataFrame(draft["derived"] or [], columns=["name", "from", "transform"])
    edited = st.data_editor(
        derived_df, key=ekey("derived"), num_rows="dynamic", hide_index=True,
        width="stretch",
        column_config={
            "from": st.column_config.SelectboxColumn(
                options=[c["name"] for c in draft["columns"] if c["include"]]),
            "transform": st.column_config.SelectboxColumn(options=TRANSFORMS)})
    new_derived = [{"name": r["name"], "from": r["from"], "transform": r["transform"]}
                   for _, r in edited.iterrows()
                   if isinstance(r["name"], str) and r["name"] and r["from"]
                   and r["transform"]]
    if new_derived != draft["derived"]:
        draft["derived"] = new_derived
        s["pricer"].types = declared_types(draft)
        names = declared_names(draft)
        types = declared_types(draft)

    st.subheader("Comparisons")
    st.caption("Levels are evaluated top down and the first that fires wins, so the "
               "order is the model. Every comparison starts at `null` and ends at "
               "`else`. Thresholds are starting values; `cpplink levels` places them "
               "from the data.")
    remove = None
    for index, comparison in enumerate(draft["comparisons"]):
        title = f"{comparison.get('name', '?')}: {', '.join(comparison['columns'])}"
        with st.expander(title, expanded=False):
            c1, c2, c3, c4 = st.columns([2, 3, 1, 1])
            comparison["name"] = c1.text_input("name", comparison.get("name", ""),
                                               key=f"cname-{index}-{ekey('c')}")
            comparison["columns"] = c2.multiselect(
                "columns", names, default=[c for c in comparison["columns"] if c in names],
                key=f"ccols-{index}-{ekey('c')}")
            comparison["term_frequency"] = c3.checkbox(
                "term frequency", comparison.get("term_frequency", False),
                key=f"ctf-{index}-{ekey('c')}")
            if c4.button("delete", key=f"cdel-{index}-{ekey('c')}"):
                remove = index
            levels = pd.DataFrame(
                [{"type": lv["type"], "threshold": lv.get("threshold"),
                  "label": lv.get("label", ""), "column": lv.get("column", "")}
                 for lv in comparison["levels"]],
                columns=["type", "threshold", "label", "column"])
            edited = st.data_editor(
                levels, key=f"clev-{index}-{ekey('c')}", num_rows="dynamic",
                hide_index=True, width="stretch",
                column_config={
                    "type": st.column_config.SelectboxColumn(options=LEVEL_TYPES,
                                                             required=True),
                    "threshold": st.column_config.NumberColumn(),
                    "column": st.column_config.SelectboxColumn(
                        options=[""] + comparison["columns"],
                        help="Only for a comparison over several string columns")})
            new_levels = []
            for _, r in edited.iterrows():
                if not isinstance(r["type"], str):
                    continue
                level = {"type": r["type"]}
                if pd.notna(r["threshold"]) and r["type"] not in ("null", "exact", "else"):
                    t = float(r["threshold"])
                    level["threshold"] = int(t) if t == int(t) else t
                if isinstance(r["label"], str) and r["label"]:
                    level["label"] = r["label"]
                if isinstance(r["column"], str) and r["column"]:
                    level["column"] = r["column"]
                new_levels.append(level)
            comparison["levels"] = new_levels
    if remove is not None:
        del draft["comparisons"][remove]
        bump()
        st.rerun()

    st.subheader("Add a comparison")
    a1, a2, a3 = st.columns([3, 3, 2])
    column = a1.selectbox("column", names, key="add-column")
    role = a2.selectbox("template", [r for r in guess.ROLES if r not in
                                     templates.NO_COMPARISON], key="add-role")
    if a3.button("Add"):
        derived_names = {d["name"]: (d["from"], d["transform"]) for d in draft["derived"]}
        comparison = templates.comparison_for(column, role, types, derived_names)
        if comparison is not None:
            if any(c.get("name") == comparison["name"] for c in draft["comparisons"]):
                comparison["name"] += f"_{role}"
            draft["comparisons"].append(comparison)
            bump()
            st.rerun()


# --- Blocking tab -------------------------------------------------------------

def blocking_tab():
    s = state()
    ds, draft = s["ds"], s["draft"]
    names = declared_names(draft)
    types = declared_types(draft)
    blockable = [n for n in names if types.get(n) not in ("double", "string_list")]
    columns_declared = schema.to_schema(draft)["columns"]

    top1, top2, top3 = st.columns([3, 2, 5])
    budget = top1.select_slider("Candidate budget, per row", BUDGETS,
                                value=s.get("budget", BUDGETS[5]),
                                help="The 20M run scored 500 candidates per row in 38 "
                                     "minutes on one thread.")
    s["budget"] = budget
    if top2.button("Draft a plan under this budget", type="primary"):
        redraft_plan(budget)
        bump()
        st.rerun()
    if s.get("plan_reasons"):
        with top3.expander("why this plan"):
            for reason in s["plan_reasons"]:
                st.write("- " + reason)

    st.caption("A pair belongs to the first source that produces it, so order matters: "
               "a source's marginal contribution depends on what sits above it. "
               "`minhash` is priced by the binary only; single-column MinHash was "
               "dominated on every dataset measured.")
    rows = []
    for source in draft["blocking"]:
        rows.append({"type": source["type"], "column": source.get("column", ""),
                     "max_frequency": source.get("max_frequency"),
                     "window": source.get("window"), "bands": source.get("bands"),
                     "rows_per_band": source.get("rows_per_band"),
                     "ngram": source.get("ngram")})
    edited = st.data_editor(
        pd.DataFrame(rows, columns=["type", "column", "max_frequency", "window", "bands",
                                    "rows_per_band", "ngram"]),
        key=ekey("blocking"), num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "type": st.column_config.SelectboxColumn(options=blocking.SOURCE_TYPES,
                                                     required=True),
            "column": st.column_config.SelectboxColumn(options=[""] + blockable),
            "max_frequency": st.column_config.NumberColumn(
                "max_frequency (rare_value)", min_value=0, step=1,
                help="Values seen more often than this never block; 0 means no cap"),
            "window": st.column_config.NumberColumn("window (sorted_neighbourhood)",
                                                    min_value=1, step=1),
            "bands": st.column_config.NumberColumn("bands (minhash)", min_value=1, step=1),
            "rows_per_band": st.column_config.NumberColumn(min_value=1, step=1),
            "ngram": st.column_config.NumberColumn(min_value=1, step=1)})
    plan = []
    for _, r in edited.iterrows():
        if not isinstance(r["type"], str):
            continue
        source = {"type": r["type"]}
        if r["type"] != "all_pairs":
            if not isinstance(r["column"], str) or not r["column"]:
                continue
            source["column"] = r["column"]
        if r["type"] == "rare_value":
            source["max_frequency"] = int(r["max_frequency"]) \
                if pd.notna(r["max_frequency"]) else 100
        if r["type"] == "sorted_neighbourhood":
            source["window"] = int(r["window"]) if pd.notna(r["window"]) else 8
        if r["type"] == "minhash":
            source["bands"] = int(r["bands"]) if pd.notna(r["bands"]) else 12
            source["rows_per_band"] = int(r["rows_per_band"]) \
                if pd.notna(r["rows_per_band"]) else 4
            source["ngram"] = int(r["ngram"]) if pd.notna(r["ngram"]) else 3
        plan.append(source)
    draft["blocking"] = plan
    if not plan:
        st.info("no source yet: draft a plan, or add a row")
        return

    priced = blocking.price_plan(s["pricer"], plan, columns_declared, ds.rows)
    labels = [source_label(src) for src in plan]
    rows = []
    for src, label, p in zip(plan, labels, priced["sources"]):
        rows.append({
            "source": label,
            "candidates": p["candidates"] if p else None,
            "share %": 100 * p["candidates"] / priced["candidate_sum"]
            if p and priced["candidate_sum"] else None,
            "rows it can see %": 100 * p["coverage"] if p else None,
            "largest group": p["largest_group"] if p else None,
            "priced": "closed form" if p else "binary only",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                 column_config={"candidates": st.column_config.NumberColumn(
                                    format="localized"),
                                "largest group": st.column_config.NumberColumn(
                                    format="localized"),
                                "share %": st.column_config.NumberColumn(format="%.1f"),
                                "rows it can see %": st.column_config.NumberColumn(
                                    format="%.1f", help="Rows the source can put in a "
                                    "pair: non-null, and under the cap")})

    total = priced["candidate_sum"]
    est = blocking.wall_estimate(total, s.get("threads", 1))
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Candidates (sum)", f"{total:,}",
              help="Sum over sources, which bounds the deduplicated union from above")
    m2.metric("Per row", f"{total / ds.rows:,.0f}" if ds.rows else "0")
    m3.metric("Reduction", f"{1 - total / priced['pair_space']:.6f}"
              if priced["pair_space"] else "0")
    m4.metric(f"Score, {s.get('threads', 1)} thread(s)",
              blocking.format_seconds(est["ceiling_s"]),
              help="At the 207 ns/candidate measured with the score ceiling on; "
                   f"about {blocking.format_seconds(est['full_s'])} without it")
    m5.metric("Enumerate", blocking.format_seconds(est["enumerate_s"]),
              help="At the 9 ns/pair measured for walking the union")
    if priced["unpriced"]:
        st.info(f"{priced['unpriced']} source(s) can only be priced by the binary")
    if total > budget * ds.rows:
        st.warning(f"over budget: {total:,} against {budget * ds.rows:,}")

    left, right = st.columns(2)
    values = [p["candidates"] if p else 0 for p in priced["sources"]]
    fig = bar_figure(labels, values, "Candidates per source", log=True,
                     line=budget * ds.rows, unit=" candidates")
    left.plotly_chart(fig, width="stretch")

    pick = right.selectbox("Source to examine", list(range(len(plan))),
                           format_func=lambda i: labels[i])
    source = plan[pick]
    resolved = blocking.resolve_column(source.get("column", ""), columns_declared) \
        if source["type"] != "all_pairs" else None
    got = s["pricer"].counts(*resolved) if resolved else None
    if got is not None and source["type"] in ("exact_value", "rare_value"):
        counts, nulls = got
        caps = blocking.default_caps(counts)
        curve = blocking.rare_curve(counts, caps)
        mark = source.get("max_frequency") if source["type"] == "rare_value" else None
        fig = curve_figure([p["cap"] for p in curve], [p["candidates"] for p in curve],
                           f"Candidates against the cap, {source['column']}",
                           "max_frequency", "candidates", mark=mark,
                           hover="cap %{x}: %{y:,} candidates<extra></extra>")
        right.plotly_chart(fig, width="stretch")
        fig = curve_figure([p["cap"] for p in curve],
                           [100 * p["coverage"] for p in curve],
                           "Rows the source can see, %", "max_frequency", "rows %",
                           mark=mark, logy=False, color=AQUA,
                           hover="cap %{x}: %{y:.1f}% of rows<extra></extra>")
        right.plotly_chart(fig, width="stretch")
        hist = blocking.group_size_histogram(counts)
        fig = go.Figure(go.Bar(x=[f"{b['low']}–{b['high']}" for b in hist],
                               y=[b["pairs"] for b in hist], marker_color=BLUE,
                               customdata=[[b["values"], b["rows"]] for b in hist],
                               hovertemplate="group size %{x}<br>%{customdata[0]:,} "
                                             "values, %{customdata[1]:,} rows<br>"
                                             "%{y:,} pairs<extra></extra>"))
        fig.update_layout(title=f"Pairs by group size, {source['column']}", height=300,
                          margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title="rows sharing the value",
                          yaxis=dict(title="pairs", type="log"))
        left.plotly_chart(fig, width="stretch")
        top = stats.top_values(ds, resolved[0], 10) if resolved[1] is None else []
        if top:
            left.caption("Largest groups: " + ", ".join(
                f"`{v}` {fmt_int(c)} rows, {fmt_int(blocking.pairs_in(c))} pairs"
                for v, c in top[:5]))
    elif got is not None and source["type"] == "sorted_neighbourhood":
        counts, nulls = got
        windows = [2, 4, 8, 16, 32, 64, 128, 256]
        curve = blocking.window_curve(ds.rows - nulls, windows)
        fig = curve_figure(windows, [p["candidates"] for p in curve],
                           f"Candidates against the window, {source['column']}",
                           "window", "candidates", mark=source.get("window"),
                           hover="window %{x}: %{y:,} candidates<extra></extra>")
        right.plotly_chart(fig, width="stretch")
    elif source["type"] == "minhash":
        right.info("MinHash is priced by the binary: bands over shingles cannot be "
                   "counted from the term frequencies.")

    st.subheader("From the binary")
    paths, why = data_paths_for_binary()
    if why:
        st.info(why)
        return
    schema_text = schema.to_json(draft)
    b1, b2, b3 = st.columns(3)
    if b1.button("Price with cpplink (exact, seconds)"):
        run_binary("explain", lambda: binary.explain_blocking(schema_text, paths))
    if b2.button("Count the deduplicated union (enumerates every pair)"):
        run_binary("union", lambda: binary.explain_blocking(schema_text, paths, count=True))
    truth = s.get("truth")
    if b3.button("Recall against the truth file", disabled=not truth):
        run_binary("recall", lambda: binary.recall(schema_text, paths, truth))
    for key in ("explain", "union"):
        if key in s["reports"]:
            report, elapsed = s["reports"][key]
            st.caption(f"cpplink explain-blocking, {elapsed:.1f} s")
            df = pd.DataFrame(report["sources"])
            st.dataframe(df, width="stretch", hide_index=True)
            line = f"sum {report['candidate_sum']:,} over a pair space of " \
                   f"{report['pair_space']:,}"
            if report.get("counted_union"):
                line += f"; union {report['candidate_union']:,} " \
                        f"({100 * report['candidate_union'] / max(report['candidate_sum'], 1):.1f}% of the sum)"
            st.write(line)
    if "recall" in s["reports"]:
        report, elapsed = s["reports"]["recall"]
        st.caption(f"cpplink recall, {elapsed:.1f} s")
        r1, r2, r3 = st.columns(3)
        r1.metric("Pair completeness", f"{100 * report['pair_completeness']:.2f}%")
        r2.metric("Pair quality", f"{100 * report['pair_quality']:.3f}%")
        r3.metric("Reduction ratio", f"{report['reduction_ratio']:.6f}")
        df = pd.DataFrame(report["sources"])
        st.dataframe(df, width="stretch", hide_index=True)
        fig = bar_figure([src["name"] for src in report["sources"]],
                         [src["first_to"] for src in report["sources"]],
                         "Truth pairs each source is the first to reach", color=AQUA,
                         unit=" pairs")
        st.plotly_chart(fig, width="stretch")


def source_label(source):
    kind = source["type"]
    if kind == "all_pairs":
        return "all pairs"
    if kind == "rare_value":
        return f"{source['column']} ≤ {source.get('max_frequency', 100)}"
    if kind == "sorted_neighbourhood":
        return f"{source['column']} window {source.get('window', 8)}"
    if kind == "minhash":
        return f"{source['column']} minhash {source.get('bands', 12)}×" \
               f"{source.get('rows_per_band', 4)}"
    return f"{source['column']} exact"


def run_binary(key, call):
    s = state()
    try:
        with st.spinner("running cpplink"):
            report, err, elapsed = call()
        s["reports"][key] = (report, elapsed)
    except binary.BinaryError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))


# --- Schema tab ---------------------------------------------------------------

def schema_tab():
    s = state()
    draft = s["draft"]
    found = schema.problems(draft)
    if found:
        for problem in found:
            st.error(problem)
    else:
        st.success("no editor-level problem; the binary's parser is the validator")
    text = schema.to_json(draft)
    c1, c2, c3 = st.columns([3, 1, 2])
    out = c1.text_input("Save as", value=os.path.splitext(s["ds"].path)[0] + "_schema.json")
    if c2.button("Save"):
        with open(out, "w") as handle:
            handle.write(text)
        st.success(f"wrote {out}")
    c3.download_button("Download", text, file_name="schema.json", mime="application/json")
    paths, why = data_paths_for_binary()
    if why:
        st.info(why)
    elif st.button("Validate with cpplink inspect"):
        try:
            with st.spinner("inspecting"):
                out_text, err, elapsed = binary.inspect(text, paths)
            st.code(out_text)
        except binary.BinaryError as exc:
            st.error(str(exc))
    st.code(text, language="json")


# --- main ---------------------------------------------------------------------

def main():
    sidebar()
    s = state()
    if "ds" not in s:
        st.title("cpplink studio")
        st.write("Open a csv or parquet file from the sidebar. The page reads it "
                 "through DuckDB, profiles every column, guesses what each holds, "
                 "drafts a schema and a blocking plan, and prices the plan before "
                 "anything runs.")
        default = sys.argv[1] if len(sys.argv) > 1 else ""
        if default and os.path.exists(default):
            load_dataset(default)
            st.rerun()
        return
    tabs = st.tabs(["Data", "Columns", "Correlation", "Comparisons", "Blocking", "Schema"])
    with tabs[0]:
        data_tab()
    with tabs[1]:
        columns_tab()
    with tabs[2]:
        correlation_tab()
    with tabs[3]:
        comparisons_tab()
    with tabs[4]:
        blocking_tab()
    with tabs[5]:
        schema_tab()


main()
