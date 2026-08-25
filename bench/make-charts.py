#!/usr/bin/env python3
"""Generate bench/charts.html from the JSONL logs.

Regenerate rather than hand-edit. Palette is the validated three-slot categorical set
(#2a78d6 / #eb6834 / #1baf7a): all checks pass in light mode, aqua sits below 3:1 contrast so the
relief rule applies and every mark carries a visible direct label.
"""
import json, pathlib, statistics, collections, html

HERE = pathlib.Path(__file__).resolve().parent


def load(name):
    p = HERE / name
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def esc(s):
    return html.escape(str(s))


# ---------------------------------------------------------------- bar chart

def bar_chart(title, subtitle, items, unit="%", series_label=None):
    """items: [(label, value, note)] — horizontal bars, one series, value labelled at the tip."""
    if not items:
        return ""
    vals = [v for _, v, _ in items]
    lo, hi = min(0, min(vals)), max(vals)
    span = (hi - lo) or 1
    row_h, gap, left, right = 34, 10, 260, 90
    width, height = 900, len(items) * (row_h + gap) + 8
    zero = left + (0 - lo) / span * (width - left - right)

    bars = []
    for i, (label, value, note) in enumerate(items):
        y = i * (row_h + gap)
        x_val = left + (value - lo) / span * (width - left - right)
        x, w = (zero, x_val - zero) if value >= 0 else (x_val, zero - x_val)
        w = max(abs(w), 1.5)
        pos = value >= 0
        # 4px rounded data-end, square at the baseline
        r = 4
        d = (f"M{x} {y} h{w-r} a{r} {r} 0 0 1 {r} {r} v{row_h-2*r} a{r} {r} 0 0 1 -{r} {r} H{x} Z"
             if pos else
             f"M{x+w} {y} H{x+r} a{r} {r} 0 0 0 -{r} {r} v{row_h-2*r} a{r} {r} 0 0 0 {r} {r} h{w-r} Z")
        label_x = (x + w + 8) if pos else (x - 8)
        anchor = "start" if pos else "end"
        bars.append(f"""
      <g class="bar-row" tabindex="0">
        <title>{esc(label)}: {value:+.1f}{unit}{(' — ' + esc(note)) if note else ''}</title>
        <text class="cat" x="{left-12}" y="{y+row_h/2+5}" text-anchor="end">{esc(label)}</text>
        <path class="bar" d="{d}" fill="var(--series-1)"/>
        <text class="val" x="{label_x}" y="{y+row_h/2+5}" text-anchor="{anchor}">{value:+.1f}{unit}</text>
      </g>""")

    return f"""
  <figure class="chart">
    <figcaption><h3>{esc(title)}</h3><p>{esc(subtitle)}</p></figcaption>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">
      <line class="axis" x1="{zero}" y1="0" x2="{zero}" y2="{height-6}"/>
      {''.join(bars)}
    </svg>
  </figure>"""


# ---------------------------------------------------------------- line chart

def line_chart(title, subtitle, series, x_label, y_label, y_unit=""):
    """series: {name: [(x, y)]} — one line per series, 2px, end-dot >= 8px with a surface ring."""
    series = {k: sorted(v) for k, v in series.items() if v}
    if not series:
        return ""
    xs = sorted({x for pts in series.values() for x, _ in pts})
    ys = [y for pts in series.values() for _, y in pts]
    ymax = max(ys) * 1.12
    ymin = 0
    W, H = 900, 380
    pad_l, pad_r, pad_t, pad_b = 70, 130, 20, 46

    def px(i):
        return pad_l + (i / max(len(xs) - 1, 1)) * (W - pad_l - pad_r)

    def py(v):
        return H - pad_b - (v - ymin) / (ymax - ymin or 1) * (H - pad_t - pad_b)

    grid, ticks = [], 5
    for t in range(ticks + 1):
        v = ymin + (ymax - ymin) * t / ticks
        y = py(v)
        grid.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}"/>'
                    f'<text class="tick" x="{pad_l-10}" y="{y+4:.1f}" text-anchor="end">{v:,.0f}</text>')
    for i, x in enumerate(xs):
        grid.append(f'<text class="tick" x="{px(i):.1f}" y="{H-pad_b+20}" text-anchor="middle">'
                    f'{x:,}</text>')

    if len(series) > 3:
        raise ValueError(f"{title}: {len(series)} series but only 3 validated palette slots; "
                         "facet instead of adding hues")
    paths, legend = [], []
    for slot, (name, pts) in enumerate(series.items(), start=1):
        colour = f"var(--series-{slot})"
        d = " ".join(f"{'M' if j == 0 else 'L'}{px(xs.index(x)):.1f} {py(y):.1f}"
                     for j, (x, y) in enumerate(pts))
        dots = "".join(
            f'<circle class="dot" cx="{px(xs.index(x)):.1f}" cy="{py(y):.1f}" r="4.5" '
            f'fill="{colour}"><title>{esc(name)} — {x_label} {x:,}: {y:,.1f}{y_unit}</title></circle>'
            for x, y in pts)
        lx, ly = px(xs.index(pts[-1][0])), py(pts[-1][1])
        paths.append(f'<path class="line" d="{d}" stroke="{colour}"/>{dots}'
                     f'<text class="endlabel" x="{lx+12:.1f}" y="{ly+4:.1f}">{pts[-1][1]:,.1f}{y_unit}</text>')
        legend.append(f'<span class="key"><i style="background:{colour}"></i>{esc(name)}</span>')

    return f"""
  <figure class="chart">
    <figcaption><h3>{esc(title)}</h3><p>{esc(subtitle)}</p></figcaption>
    <div class="legend">{''.join(legend)}</div>
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">
      {''.join(grid)}
      <line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{H-pad_b}"/>
      <line class="axis" x1="{pad_l}" y1="{H-pad_b}" x2="{W-pad_r}" y2="{H-pad_b}"/>
      <text class="axlabel" x="{(W-pad_r+pad_l)/2:.0f}" y="{H-6}" text-anchor="middle">{esc(x_label)}</text>
      <text class="axlabel" transform="rotate(-90 16 {H/2:.0f})" x="16" y="{H/2:.0f}" text-anchor="middle">{esc(y_label)}</text>
      {''.join(paths)}
    </svg>
  </figure>"""


def table(headers, rows, caption):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<details class="tableview"><summary>{esc(caption)}</summary>'
            f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></details>')


# ---------------------------------------------------------------- assembly

def main():
    depth = load("results.jsonl")
    gates = load("results-gates.jsonl")
    head = load("results-fp4-headline.jsonl")
    sustained = load("results-fp4-spec.jsonl")
    sdepth = load("results-spec-depth.jsonl")

    parts = []

    # 1. gate ablations, biggest first
    gate_items = sorted(({"lbl": f'{r["gate"]} · {r["quant"]}', "v": r["delta_pct"],
                          "note": r["model"]} for r in gates),
                        key=lambda d: -d["v"])
    parts.append(bar_chart(
        "Vulkan gate ablations",
        "pp2048 at ubatch 2048, each gate on vs off in one binary. Three models, three quant families.",
        [(d["lbl"], d["v"], d["note"]) for d in gate_items]))
    parts.append(table(["gate", "model", "quant", "off t/s", "on t/s", "delta %"],
                       [[r["gate"], r["model"], r["quant"], f'{r["off"]:.1f}', f'{r["on"]:.1f}',
                         f'{r["delta_pct"]:+.1f}'] for r in gates],
                       "Gate ablations — table view"))

    # 2. speculative arms, both power profiles
    def spec_items(rows, key="predicted_per_second"):
        out = []
        for r in rows:
            if r.get(key) and r.get("workload") in ("prose", "json"):
                out.append((f'{r["policy"]} · {r["workload"]}', r[key],
                            f'acceptance {r["accept_pct"]:.0f}%' if r.get("accept_pct") else "no draft"))
        return out

    if head:
        items = spec_items(head)
        vals = [(l, v, n) for l, v, n in items]
        parts.append(bar_chart(
            "Speculative decoding — FP4 target, FP4 draft (peak power profile)",
            "Qwen3.8-27B-ROCmFP4-FAST + DFlash2 FP4 sidecar, greedy, 300 tokens, depth 0. "
            "Absolute t/s, higher is better.",
            vals, unit=" t/s"))
        parts.append(table(["policy", "workload", "t/s", "acceptance", "power"],
                           [[r["policy"], r["workload"], f'{r["predicted_per_second"]:.2f}',
                             f'{r["accept_pct"]:.0f}%' if r.get("accept_pct") else "—",
                             r.get("power", "—")] for r in head],
                           "Speculative arms — table view"))

    # 3. fork vs mainline against depth
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in depth:
        test = f'pp{r["n_prompt"]}' if r.get("n_prompt") else f'tg{r["n_gen"]}'
        by[(r["model_label"], test)][(r["n_depth"], r["build"])].append(r["avg_ts"])
    for test, label, unit in (("pp2048", "Prefill", " t/s"), ("tg64", "Generation", " t/s")):
        series = {}
        for (model, t), cells in by.items():
            if t != test:
                continue
            pts = []
            for depth_v in sorted({d for d, _ in cells}):
                a = cells.get((depth_v, "mainline"), [])
                b = cells.get((depth_v, "fork"), [])
                if a and b:
                    pts.append((depth_v, 100 * (statistics.mean(b) / statistics.mean(a) - 1)))
            if pts:
                series[model] = pts
        if series:
            parts.append(line_chart(
                f"{label}: fork vs pinned upstream, against context depth",
                "Percent difference against upstream 95b8e33e1, the exact commit this fork merged. "
                "ubatch 512, flash attention on, palindrome-ordered arms.",
                series, "context depth (tokens)", f"{label} delta vs upstream (%)", "%"))

    # 4. adaptive against depth
    #
    # Two exclusions, both about the measurement rather than the result:
    #  * degenerate rows (the model stopped early) carry a meaningless rate — one MTP prose cell at
    #    16384 came back 0.00 t/s from a run that predated the degeneracy guard.
    #  * depth 0 is dropped from the *prefill* chart only. With no filler the prompt is ~40 tokens,
    #    so prompt_per_second is dominated by fixed overhead and reads far below the deeper points,
    #    which would draw prefill as rising with depth. Generation at depth 0 is fine and kept.
    def usable(r, metric):
        v = r.get(metric)
        if not v or r.get("degenerate"):
            return False
        if (r.get("predicted_n") or 0) < 270:
            return False
        return not (metric == "prompt_per_second" and r["depth"] == 0)

    if sdepth:
        workloads = sorted({r["workload"] for r in sdepth})
        for metric, lbl, unit in (("prompt_per_second", "Prefill", " t/s"),
                                  ("predicted_per_second", "Generation", " t/s")):
          for workload in workloads:
            # Facet by workload rather than colouring four series. Two methods per chart keeps the
            # palette inside its validated slots, and prose and structured output are different
            # enough that overlaying them obscures both.
            series = collections.defaultdict(list)
            for r in sdepth:
                if r["workload"] == workload and usable(r, metric):
                    series[r["policy"]].append((r["depth"], r[metric]))
            if series:
                parts.append(line_chart(
                    f"Adaptive drafting — {lbl.lower()} against context depth ({workload})",
                    "Qwen3.8-27B UD-Q4_K_XL. MTP uses the target's own nextn layers; DFlash2 uses "
                    "the z-lab Q8_0 sidecar. Both adaptive, n_max 7, n_min 3. Degenerate runs "
                    "excluded" + ("; depth 0 omitted, its ~40-token prompt measures overhead rather "
                    "than prefill" if metric == "prompt_per_second" else "") + ".",
                    dict(series), "context depth (tokens)", f"{lbl} ({unit.strip()})", unit))
    else:
        parts.append('<figure class="chart missing"><figcaption><h3>Adaptive drafting against '
                     'context depth</h3><p>Not yet measured. Run '
                     '<code>python3 bench/spec_bench.py --preset kquant --depths 0,4096,16384,32768</code>'
                     ' and regenerate.</p></figcaption></figure>')

    css = """
    :root { color-scheme: light dark; }
    .viz-root {
      color-scheme: light;
      --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
      --text-muted: #767570; --grid: #e7e6e2;
      --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
    }
    @media (prefers-color-scheme: dark) {
      :root:where(:not([data-theme="light"])) .viz-root {
        color-scheme: dark;
        --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
        --text-muted: #96958c; --grid: #33322f;
        --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
      }
    }
    :root[data-theme="dark"] .viz-root {
      color-scheme: dark;
      --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
      --text-muted: #96958c; --grid: #33322f;
      --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    }
    body { margin: 0; background: var(--surface-1); color: var(--text-primary);
           font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
    .viz-root { max-width: 1000px; margin: 0 auto; padding: 40px 24px 80px; }
    h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
    .lede { color: var(--text-secondary); margin: 0 0 8px; max-width: 68ch; }
    .caveat { color: var(--text-muted); font-size: 13px; max-width: 68ch;
              border-left: 2px solid var(--grid); padding-left: 12px; margin: 16px 0 40px; }
    .chart { margin: 0 0 52px; }
    .chart figcaption h3 { font-size: 17px; margin: 0 0 4px; }
    .chart figcaption p { margin: 0 0 14px; color: var(--text-secondary); font-size: 13.5px;
                          max-width: 72ch; }
    svg { width: 100%; height: auto; overflow: visible; }
    .axis { stroke: var(--grid); stroke-width: 1; }
    .grid { stroke: var(--grid); stroke-width: 1; }
    .tick, .cat { fill: var(--text-secondary); font-size: 12.5px; }
    .axlabel { fill: var(--text-muted); font-size: 12px; }
    .val, .endlabel { fill: var(--text-primary); font-size: 12.5px; font-weight: 600; }
    .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
    .dot { stroke: var(--surface-1); stroke-width: 2; }
    .bar-row:hover .bar, .bar-row:focus .bar { opacity: 0.82; }
    .bar-row:focus { outline: none; }
    .bar-row:focus .cat { fill: var(--text-primary); font-weight: 600; }
    .legend { display: flex; gap: 18px; flex-wrap: wrap; margin: 0 0 10px; font-size: 13px;
              color: var(--text-secondary); }
    .key { display: inline-flex; align-items: center; gap: 7px; }
    .key i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
    .tableview { margin: -34px 0 52px; font-size: 13px; }
    .tableview summary { cursor: pointer; color: var(--text-muted); }
    table { border-collapse: collapse; margin-top: 12px; width: 100%; }
    th, td { text-align: left; padding: 5px 12px 5px 0; border-bottom: 1px solid var(--grid); }
    th { color: var(--text-muted); font-weight: 600; }
    .missing figcaption p { color: var(--text-muted); }
    code { font-size: 12.5px; background: var(--grid); padding: 1px 5px; border-radius: 3px; }
    """

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llama.cpp Strix Halo fork — benchmark results</title>
<style>{css}</style></head>
<body><main class="viz-root">
  <h1>Strix Halo Vulkan fork — benchmark results</h1>
  <p class="lede">Radeon 8060S (gfx1151, RDNA 3.5), RADV / Mesa 26.0.8. Upstream pinned at
  <code>95b8e33e1</code>, the exact commit this fork merged, so deltas are this fork's changes and
  not upstream drift.</p>
  <p class="caveat"><strong>Read deltas across charts, not absolutes.</strong> The fork-vs-upstream
  and gate figures are ratios measured within one session on one power profile with interleaved
  arms — those hold. Absolute t/s depends on the power profile, and the speculative chart was taken
  on a short high-power burst (79 °C, 115 W) that this chassis cannot sustain all day. Every
  generation figure carries its context depth: these models declare 262144 context and generation at
  depth is roughly a third of its depth-0 value.</p>
  {''.join(parts)}
</main></body></html>"""

    (HERE / "charts.html").write_text(doc)
    print(f"wrote {HERE/'charts.html'} ({len(doc)//1024} KB)")


main()
