# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Collect a single-threaded scale sweep and draw the two scaling figures.

Reads the per-stage logs written by ``run_cpplink_scale.sh`` for each tag in the
sweep, writes one JSON record per size, and emits two SVGs: wall time against
candidate pairs, and peak resident memory against candidate pairs.

Figures are drawn with matplotlib where it is installed and by a hand-written SVG
backend where it is not, so the benchmark keeps working in an environment that
carries no plotting library. Both draw the same figure: the same colours, chosen
to read on a light and a dark page, the same 1-2-5 ticks, the same annotations.
``--backend`` forces one or the other, which is how the two are compared.

Usage:
    python3 bench/scale/plot_scale.py <workdir> --tags s250k s500k ... \\
        --json bench/scale/single_thread.json --outdir docs/img
    python3 bench/scale/plot_scale.py --from-json bench/scale/single_thread.json \\
        --outdir docs/img
"""
import argparse
import json
import math
import os
import re


def read(path):
    try:
        with open(path) as handle:
            return handle.read()
    except FileNotFoundError:
        return ""


def number(text, pattern, cast=float):
    found = re.search(pattern, text)
    if not found:
        return None
    return cast(found.group(1).replace(",", ""))


def stage(workdir, tag, name):
    """Wall seconds and peak RSS in bytes for one timed stage."""
    timing = read(os.path.join(workdir, f"{tag}.{name}.time"))
    return {
        "seconds": number(timing, r"([\d.]+) real"),
        "rss_bytes": number(timing, r"(\d+)\s+maximum resident set size", int),
    }


def collect(workdir, tag):
    blocking = read(os.path.join(workdir, f"{tag}.blocking.log"))
    predict = read(os.path.join(workdir, f"{tag}.predict.log"))
    cluster = read(os.path.join(workdir, f"{tag}.cluster.log"))
    stages = {n: stage(workdir, tag, n) for n in ("blocking", "estimate", "predict", "cluster")}
    pipeline = [stages[n] for n in ("estimate", "predict", "cluster")]
    record = {
        "tag": tag,
        "records": number(blocking, r"Records\s+([\d,]+)", int),
        "candidate_pairs": number(predict, r"Candidates\s+([\d,]+)", int),
        "sum_over_sources": number(blocking, r"Sum over sources\s+([\d,]+)", int),
        "pairs_unblocked": number(blocking, r"Pairs unblocked\s+([\d,]+)", int),
        "edges": number(predict, r"Edges\s+([\d,]+)", int),
        "candidates_per_second": number(predict, r"\(([\d]+) candidates/s\)", int),
        "skipped_share": number(predict, r"skipped\s+[\d,]+\s+([\d.]+)%"),
        "f1": number(cluster, r"f1\s+([\d.]+)"),
        "precision": number(cluster, r"precision\s+([\d.]+)"),
        "recall": number(cluster, r"recall\s+([\d.]+)"),
        "stages": stages,
        "wall_seconds": sum(s["seconds"] for s in pipeline if s["seconds"]),
        "peak_rss_bytes": max((s["rss_bytes"] or 0) for s in pipeline),
    }
    gen = read(os.path.join(workdir, f"{tag}.gen.time"))
    record["gen_seconds"] = number(gen, r"([\d.]+) real")
    parquet = os.path.join(workdir, f"{tag}.parquet")
    if os.path.exists(parquet):
        record["parquet_bytes"] = os.path.getsize(parquet)
    return record


def collect_splink(workdir, tag, records, candidate_pairs):
    """One splink run, from the timings.json its harness writes.

    Peak resident memory comes from the same /usr/bin/time -l wrapper cpplink's
    stages are measured with, so the two tools' memory columns mean one thing.
    """
    path = os.path.join(workdir, f"splink_{tag}", "timings.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        report = json.load(handle)
    timing = read(os.path.join(workdir, f"splink_{tag}.time"))
    return {
        "tag": tag,
        "records": records,
        "candidate_pairs": candidate_pairs,
        "edges": report.get("edges"),
        "wall_seconds": report.get("pipeline_seconds"),
        "peak_rss_bytes": number(timing, r"(\d+)\s+maximum resident set size", int)
        or report.get("peak_rss_bytes"),
        "peak_spill_bytes": report.get("peak_spill_bytes"),
        "stages": {k: {"seconds": v} for k, v in report.get("stages", {}).items()},
        "f1": report.get("f1"),
    }


# --- what both backends draw ----------------------------------------------

WIDTH, HEIGHT = 720, 420
LEFT, RIGHT, TOP, BOTTOM = 78, 24, 76, 62
AXIS = "#8a8f98"
TEXT = "#8a8f98"
TITLE = "#9aa0a8"
SERIES = ["#6366f1", "#d97706", "#0d9488", "#be185d"]
GUIDE = "#be185d"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

TIME_TITLE = "Wall time against candidate pairs, one thread"
MEMORY_TITLE = "Peak memory against candidate pairs, one thread"
PAIRS_LABEL = "candidate pairs (log scale)"


def si(value):
    for limit, suffix in ((1e9, "bn"), (1e6, "M"), (1e3, "k")):
        if value >= limit:
            scaled = value / limit
            text = f"{scaled:.0f}" if scaled >= 10 else f"{scaled:.1f}"
            return text + suffix
    return f"{value:g}"


def seconds_label(value):
    if value >= 1:
        return f"{value:,.0f} s"
    return f"{value:g} s"


def decades(lo, hi, steps=(1, 2, 5)):
    """1-2-5 ticks covering [lo, hi]."""
    ticks = []
    power = math.floor(math.log10(lo))
    while 10**power <= hi:
        for step in steps:
            value = step * 10**power
            if lo * 0.9 <= value <= hi * 1.1:
                ticks.append(value)
        power += 1
    return ticks


def log_limits(values, pad):
    """The axis range a log axis gets: the data, padded by a share of its span."""
    lo, hi = math.log10(min(values)), math.log10(max(values))
    span = pad * (hi - lo)
    return 10 ** (lo - span), 10 ** (hi + span)


def linear_guide(rows):
    """The dashed reference line: exactly linear in pairs, through the largest run.

    Anchored on the largest run and extended back, so the fixed cost of load and
    estimation shows up as the small runs sitting above the line rather than as an
    apparently sublinear curve.
    """
    first, last = rows[0], rows[-1]
    slope = last["wall_seconds"] / last["candidate_pairs"]
    return [
        (first["candidate_pairs"], slope * first["candidate_pairs"]),
        (last["candidate_pairs"], last["wall_seconds"]),
    ]


def spilled(rows):
    return [r for r in rows if r.get("peak_spill_bytes")]


def memory_values(tracks):
    """Every quantity the memory figure's y axis has to hold."""
    values = [r["peak_rss_bytes"] / 1e9 for t in tracks for r in t["rows"]]
    for track in tracks:
        values += [r["peak_spill_bytes"] / 1e9 for r in spilled(track["rows"])]
    return values


def spill_note(row):
    return f'{row["peak_spill_bytes"] / 1e9:.1f} GB of scratch'


SPILL_LABEL = "splink peak scratch disk"


# --- the matplotlib backend, used where it is installed --------------------


def pyplot():
    """pyplot configured to match the hand-written figures, or None if absent."""
    try:
        import matplotlib
    except ImportError:
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "none",  # keep text as text, so the page can restyle it
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.figsize": (WIDTH / 100, HEIGHT / 100),
            "figure.dpi": 100,
            "savefig.transparent": True,  # so the page's own background shows through
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": AXIS,
            "axes.titlecolor": TITLE,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    return plt


def mpl_axes(plt, title, xlabel, ylabel):
    figure, axes = plt.subplots()
    axes.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=38)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale("log")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_alpha(0.5)
    axes.grid(True, color=AXIS, alpha=0.16, linewidth=1)
    axes.set_axisbelow(True)
    axes.tick_params(length=0)
    return figure, axes


def mpl_ticks(axes, xticks, yticks, yfmt):
    axes.set_xticks(xticks)
    axes.set_xticklabels([si(v) for v in xticks])
    axes.set_yticks(yticks)
    axes.set_yticklabels([yfmt(v) for v in yticks])
    axes.minorticks_off()


def mpl_legend(axes):
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2 if len(labels) > 2 else len(labels),
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=1.6,
    )


def mpl_note(axes, x, y, text, dx=0, dy=9, ha="center", colour=None):
    axes.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        ha=ha,
        fontsize=11,
        color=colour or TEXT,
    )


def mpl_save(figure, path):
    figure.savefig(path, format="svg", bbox_inches="tight")
    figure.clf()


def mpl_time_figure(plt, tracks, path):
    pairs = [r["candidate_pairs"] for t in tracks for r in t["rows"]]
    times = [r["wall_seconds"] for t in tracks for r in t["rows"]]
    figure, axes = mpl_axes(plt, TIME_TITLE, PAIRS_LABEL, "wall seconds (log scale)")
    axes.set_yscale("log")
    axes.set_xlim(*log_limits(pairs, 0.06))
    axes.set_ylim(*log_limits(times, 0.12))
    for track in tracks:
        rows = track["rows"]
        axes.plot(
            [r["candidate_pairs"] for r in rows],
            [r["wall_seconds"] for r in rows],
            "-o",
            color=track["colour"],
            linewidth=2,
            markersize=5,
            label=track["label"],
        )
    guide = linear_guide(tracks[0]["rows"])
    axes.plot(
        [x for x, _ in guide],
        [y for _, y in guide],
        color=GUIDE,
        linewidth=2,
        linestyle=(0, (5, 4)),
        label="linear in pairs",
    )
    mpl_ticks(axes, decades(min(pairs), max(pairs)), decades(min(times), max(times)),
              seconds_label)
    for row in tracks[0]["rows"]:
        mpl_note(axes, row["candidate_pairs"], row["wall_seconds"],
                 f'{si(row["records"])} rows')
    for track in tracks[1:]:
        end = track["rows"][-1]
        if track.get("note"):
            mpl_note(axes, end["candidate_pairs"], end["wall_seconds"], track["note"],
                     dx=-4, dy=12, ha="right", colour=track["colour"])
    mpl_legend(axes)
    mpl_save(figure, path)


def mpl_memory_figure(plt, tracks, path):
    pairs = [r["candidate_pairs"] for t in tracks for r in t["rows"]]
    values = memory_values(tracks)
    figure, axes = mpl_axes(plt, MEMORY_TITLE, PAIRS_LABEL, "peak resident set size (GB)")
    axes.set_xlim(*log_limits(pairs, 0.06))
    axes.set_ylim(0, max(values) * 1.2)
    for track in tracks:
        rows = track["rows"]
        axes.plot(
            [r["candidate_pairs"] for r in rows],
            [r["peak_rss_bytes"] / 1e9 for r in rows],
            "-o",
            color=track["colour"],
            linewidth=2,
            markersize=5,
            label=track["label"],
        )
        for row in spilled(rows):
            axes.plot(
                [row["candidate_pairs"]],
                [row["peak_spill_bytes"] / 1e9],
                "o",
                markerfacecolor="none",
                markeredgecolor=GUIDE,
                markeredgewidth=2,
                markersize=9,
                label=SPILL_LABEL,
            )
            mpl_note(axes, row["candidate_pairs"], row["peak_spill_bytes"] / 1e9,
                     spill_note(row), colour=GUIDE)
    top = int(max(values)) + 2
    mpl_ticks(axes, decades(min(pairs), max(pairs)),
              [t for t in range(0, top) if t <= axes.get_ylim()[1]], lambda v: f"{v:g}")
    main_rows = tracks[0]["rows"]
    for row in (main_rows[0], main_rows[-1]):
        mpl_note(axes, row["candidate_pairs"], row["peak_rss_bytes"] / 1e9,
                 f'{row["peak_rss_bytes"] / 1e9:.2f} GB',
                 ha="right" if row is main_rows[-1] else "left")
    mpl_legend(axes)
    mpl_save(figure, path)


# --- the fallback: the smallest SVG plotter that draws a readable figure ---


class Plot:
    """A two-axis plot; either axis is log or linear."""

    def __init__(self, title, xlabel, ylabel, log_x=True, log_y=True):
        self.parts = []
        self.legend = []
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.log_x = log_x
        self.log_y = log_y

    def frame(self, xs, ys):
        if self.log_x:
            lo, hi = log_limits(xs, 0.06)
            self.x0, self.x1 = math.log10(lo), math.log10(hi)
        else:
            self.x0, self.x1 = 0.0, max(xs) * 1.06
        if self.log_y:
            lo, hi = log_limits(ys, 0.12)
            self.y0, self.y1 = math.log10(lo), math.log10(hi)
        else:
            self.y0, self.y1 = 0.0, max(ys) * 1.2

    def px(self, value):
        value = math.log10(value) if self.log_x else value
        span = self.x1 - self.x0
        return LEFT + (value - self.x0) / span * (WIDTH - LEFT - RIGHT)

    def py(self, value):
        value = math.log10(value) if self.log_y else value
        span = self.y1 - self.y0
        return HEIGHT - BOTTOM - (value - self.y0) / span * (HEIGHT - TOP - BOTTOM)

    def grid(self, xticks, yticks, xfmt, yfmt):
        for value in xticks:
            x = self.px(value)
            self.parts.append(
                f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{HEIGHT - BOTTOM}" '
                f'stroke="{AXIS}" stroke-opacity="0.16"/>'
            )
            self.parts.append(
                f'<text x="{x:.1f}" y="{HEIGHT - BOTTOM + 18}" fill="{TEXT}" '
                f'font-size="12" font-family="{FONT}" text-anchor="middle">{xfmt(value)}</text>'
            )
        for value in yticks:
            y = self.py(value)
            self.parts.append(
                f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" '
                f'stroke="{AXIS}" stroke-opacity="0.16"/>'
            )
            self.parts.append(
                f'<text x="{LEFT - 10}" y="{y + 4:.1f}" fill="{TEXT}" font-size="12" '
                f'font-family="{FONT}" text-anchor="end">{yfmt(value)}</text>'
            )

    def line(self, points, colour, label, dash=None):
        path = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in points)
        style = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<polyline points="{path}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linejoin="round"{style}/>'
        )
        if not dash:
            for x, y in points:
                self.parts.append(
                    f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="3.5" '
                    f'fill="{colour}"/>'
                )
        self.legend.append((colour, label, bool(dash)))

    def scatter(self, points, colour, label):
        """Hollow markers, for a series with too few points to draw as a line."""
        for x, y in points:
            self.parts.append(
                f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="5" '
                f'fill="none" stroke="{colour}" stroke-width="2"/>'
            )
        self.legend.append((colour, label, True))

    def annotate(self, x, y, text, dx=0, dy=-12, anchor="middle", colour=None):
        self.parts.append(
            f'<text x="{self.px(x) + dx:.1f}" y="{self.py(y) + dy:.1f}" '
            f'fill="{colour or TEXT}" '
            f'font-size="11" font-family="{FONT}" text-anchor="{anchor}">{text}</text>'
        )

    def render(self):
        head = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{self.title}">',
            f'<text x="{LEFT}" y="24" fill="{TITLE}" font-size="14" font-family="{FONT}" '
            f'font-weight="600">{self.title}</text>',
        ]
        axes = [
            f'<line x1="{LEFT}" y1="{HEIGHT - BOTTOM}" x2="{WIDTH - RIGHT}" '
            f'y2="{HEIGHT - BOTTOM}" stroke="{AXIS}" stroke-opacity="0.5"/>',
            f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{HEIGHT - BOTTOM}" '
            f'stroke="{AXIS}" stroke-opacity="0.5"/>',
            f'<text x="{(LEFT + WIDTH - RIGHT) / 2:.0f}" y="{HEIGHT - 16}" fill="{TEXT}" '
            f'font-size="12" font-family="{FONT}" text-anchor="middle">{self.xlabel}</text>',
            f'<text x="16" y="{(TOP + HEIGHT - BOTTOM) / 2:.0f}" fill="{TEXT}" font-size="12" '
            f'font-family="{FONT}" text-anchor="middle" transform="rotate(-90 16 '
            f'{(TOP + HEIGHT - BOTTOM) / 2:.0f})">{self.ylabel}</text>',
        ]
        keys = []
        x, y = LEFT + 6, TOP - 32
        for colour, label, dashed in self.legend:
            width = 34 + 7.2 * len(label)
            if x + width > WIDTH - RIGHT:          # wrap rather than run off the edge
                x, y = LEFT + 6, y + 18
            dash = ' stroke-dasharray="5 4"' if dashed else ""
            keys.append(
                f'<line x1="{x}" y1="{y}" x2="{x + 18}" y2="{y}" '
                f'stroke="{colour}" stroke-width="2"{dash}/>'
            )
            keys.append(
                f'<text x="{x + 24}" y="{y + 4}" fill="{TEXT}" font-size="12" '
                f'font-family="{FONT}">{label}</text>'
            )
            x += width
        return "\n".join(head + axes + self.parts + keys + ["</svg>"]) + "\n"


def svg_time_figure(tracks, path):
    """Wall time against candidate pairs, one line per tool and schema."""
    pairs = [r["candidate_pairs"] for t in tracks for r in t["rows"]]
    times = [r["wall_seconds"] for t in tracks for r in t["rows"]]
    plot = Plot(TIME_TITLE, PAIRS_LABEL, "wall seconds (log scale)", log_y=True)
    plot.frame(pairs, times)
    plot.grid(
        decades(min(pairs), max(pairs)),
        decades(min(times), max(times)),
        si,
        seconds_label,
    )
    for track in tracks:
        rows = track["rows"]
        plot.line(
            [(r["candidate_pairs"], r["wall_seconds"]) for r in rows],
            track["colour"],
            track["label"],
        )
    plot.line(linear_guide(tracks[0]["rows"]), GUIDE, "linear in pairs", dash="5 4")
    for row in tracks[0]["rows"]:
        plot.annotate(
            row["candidate_pairs"], row["wall_seconds"], f'{si(row["records"])} rows', dy=-13
        )
    for track in tracks[1:]:
        end = track["rows"][-1]
        if track.get("note"):
            plot.annotate(
                end["candidate_pairs"],
                end["wall_seconds"],
                track["note"],
                dx=-4,
                dy=-12,
                anchor="end",
                colour=track["colour"],
            )
    with open(path, "w") as handle:
        handle.write(plot.render())


def svg_memory_figure(tracks, path):
    """Peak memory against candidate pairs: the flatness is the design claim."""
    pairs = [r["candidate_pairs"] for t in tracks for r in t["rows"]]
    values = memory_values(tracks)
    plot = Plot(MEMORY_TITLE, PAIRS_LABEL, "peak resident set size (GB)", log_y=False)
    plot.frame(pairs, values)
    yticks = [t for t in range(0, int(max(values)) + 2) if t <= plot.y1]
    plot.grid(decades(min(pairs), max(pairs)), yticks, si, lambda v: f"{v:g}")
    for track in tracks:
        rows = track["rows"]
        plot.line(
            [(r["candidate_pairs"], r["peak_rss_bytes"] / 1e9) for r in rows],
            track["colour"],
            track["label"],
        )
        if spilled(rows):
            plot.scatter(
                [
                    (r["candidate_pairs"], r["peak_spill_bytes"] / 1e9)
                    for r in spilled(rows)
                ],
                GUIDE,
                SPILL_LABEL,
            )
            for row in spilled(rows):
                plot.annotate(
                    row["candidate_pairs"],
                    row["peak_spill_bytes"] / 1e9,
                    spill_note(row),
                    dy=-14,
                    colour=GUIDE,
                )
    main_rows = tracks[0]["rows"]
    for row in (main_rows[0], main_rows[-1]):
        plot.annotate(
            row["candidate_pairs"],
            row["peak_rss_bytes"] / 1e9,
            f'{row["peak_rss_bytes"] / 1e9:.2f} GB',
            dy=-13,
            anchor="end" if row is main_rows[-1] else "start",
        )
    with open(path, "w") as handle:
        handle.write(plot.render())


def draw(tracks, outdir, backend):
    """Draw both figures, and say which backend drew them."""
    plt = pyplot() if backend in ("auto", "matplotlib") else None
    if plt is None and backend == "matplotlib":
        raise SystemExit("--backend matplotlib: matplotlib is not installed")
    os.makedirs(outdir, exist_ok=True)
    time_path = os.path.join(outdir, "scale-time.svg")
    memory_path = os.path.join(outdir, "scale-memory.svg")
    if plt is not None:
        mpl_time_figure(plt, tracks, time_path)
        mpl_memory_figure(plt, tracks, memory_path)
        return "matplotlib"
    svg_time_figure(tracks, time_path)
    svg_memory_figure(tracks, memory_path)
    return "built-in SVG"


def build_tracks(rows, parity, splink):
    tracks = [
        {"label": "cpplink, own 10-column schema", "colour": SERIES[0], "rows": rows}
    ]
    if parity:
        tracks.append(
            {"label": "cpplink, shared schema", "colour": SERIES[2], "rows": parity}
        )
    if splink:
        tracks.append(
            {
                "label": "splink, shared schema",
                "colour": SERIES[1],
                "rows": splink,
                "note": "out of scratch disk above this",
            }
        )
    return tracks


def usable(rows):
    rows = [r for r in rows if r.get("candidate_pairs") and r.get("wall_seconds")]
    rows.sort(key=lambda r: r["records"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", nargs="?")
    parser.add_argument("--tags", nargs="+", default=[])
    parser.add_argument("--json")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--from-json",
        help="redraw the figures from a sweep this script already collected, without "
        "the run logs, which is how a figure is restyled after the logs are gone",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "matplotlib", "svg"),
        default="auto",
        help="auto uses matplotlib where it is installed and the built-in SVG writer "
        "where it is not",
    )
    parser.add_argument(
        "--parity-dir",
        help="workdir of the shared-schema runs, the cut-down model splink can be given "
        "identically; see bench/scale/make_schema.py",
    )
    parser.add_argument("--parity-tags", nargs="*", default=[])
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json) as handle:
            sweep = json.load(handle)
        rows = usable(sweep.get("runs", []))
        parity = usable(sweep.get("parity", {}).get("cpplink", []))
        splink = usable(sweep.get("parity", {}).get("splink", []))
    else:
        if not args.workdir or not args.tags or not args.json:
            parser.error("collecting a sweep needs a workdir, --tags and --json")
        rows = usable([collect(args.workdir, tag) for tag in args.tags])
        parity, splink = [], []
        if args.parity_dir:
            parity = usable([collect(args.parity_dir, tag) for tag in args.parity_tags])
            by_tag = {r["tag"]: r for r in parity}
            for tag in args.parity_tags:
                base = by_tag.get(tag)
                if not base:
                    continue
                run = collect_splink(
                    args.parity_dir, tag, base["records"], base["candidate_pairs"]
                )
                if run:
                    splink.append(run)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "threads": 1,
                    "runs": rows,
                    "parity": {"cpplink": parity, "splink": splink},
                },
                handle,
                indent=2,
            )

    tracks = build_tracks(rows, parity, splink)
    print(f"drawn with {draw(tracks, args.outdir, args.backend)}")
    for track in tracks:
        print(track["label"])
        for row in track["rows"]:
            print(
                f'  {row["tag"]:>6}  {row["records"]:>10,} rows  '
                f'{row["candidate_pairs"]:>15,} pairs  '
                f'{row["wall_seconds"]:8.1f} s  '
                f'{row["peak_rss_bytes"] / 1e9:5.2f} GB  '
                f'spill {(row.get("peak_spill_bytes") or 0) / 1e9:5.2f} GB  '
                f'F1 {row.get("f1")}'
            )


if __name__ == "__main__":
    main()
