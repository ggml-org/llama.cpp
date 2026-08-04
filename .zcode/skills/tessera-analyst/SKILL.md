---
name: tessera-analyst
description: >-
  Analyze the alphaevolve run artifacts in this repo (.zcode/alphaevolve/) and
  produce a curated research deliverable: a navigable PDF report (via the pdf
  skill) plus a narrative video script. Use when the user asks to "analyze the
  runs", "summarize tessera research", "make a research report", "explain what
  tessera did", "write a video script about the work", or similar. The data
  layer is scripts/alphaevolve-metrics.py; this skill is the workflow and
  judgment layer that interprets it and renders deliverables.
---

# Tessera Analyst

Turn the raw artifacts of alphaevolve runs into a story a human can act on:
which waves moved which metrics, what bugs were found (and fixed, or not),
what is still open - then ship two artifacts that tell that story.

## What this skill is, and is not

This skill IS: a gather-interpret-render workflow. It reads the run ledgers
and findings, decides what matters, and produces (1) a PDF research report
and (2) a narrative video script.

This skill is NOT: a way to edit the runs themselves. The ledgers and
findings are read-only inputs. Never write to `.zcode/alphaevolve/` under
this skill.

## Hard rules (non-negotiable)

- **No fabrication.** Every numeric claim in either deliverable must trace to
  a row in the metrics frame or a findings entry. Cite the run + metric key,
  or the finding ref. If a number is not in the data, it does not go in the
  report.
- **Honesty turn is required.** Both deliverables must cover what did NOT
  work, what is still open, and any non-reproductions. A report that only
  celebrates is a failed deliverable.
- **Charts follow the pdf skill.** Do not hardcode hex colors or pick them by
  feel. Run the pdf skill's `palette.generate` first; derive every chart
  color from that palette. See `typesetting/charts.md` in the pdf skill for
  the anti-overlap and spacing rules.
- **Ship both artifacts together.** The PDF report and the video script are a
  pair. The script references the report by page; the report contains the
  charts the script calls out.

## Phase 1 - Gather (data + context)

Run the data layer to produce tidy frames:

```bash
python3 scripts/alphaevolve-metrics.py --format csv --out /tmp/tessera-report/
python3 scripts/alphaevolve-metrics.py   # also print the table for a quick scan
```

This writes `metrics_long.csv` (one row per run x gene x metric, bucketed into
families: rss / tps / correctness / pct / bug) and `findings.csv` (one row per
finding: ts, run, category, severity, status, summary, source, ref).

Then read the human-written context for narrative color (the script has the
numbers; these have the *why*):

- `docs/findings-*.md` - curated findings log with reproduction notes
- `docs/audit-*.md` - prior audit context
- `.zcode/alphaevolve/<run>/integration/docs/research-*.md` - per-run research

## Phase 2 - Interpret (find the story)

Before rendering anything, decide the narrative. Scan the frames and answer:

1. **Headline metric.** Which metric moved the most across waves? (e.g. paged-
   vs-flash correctness climbing 1% -> 100% across w4 -> w5 is a stronger
   story than a 2 MiB RSS delta.) This is the report's through-line.
2. **Bug arc.** Which findings are fixed-on-main vs open vs confirmed-non-repro?
   The fixed ones with commit SHAs are the "we found and fixed real bugs" beat.
3. **Open questions.** What is still unresolved? These go in the honesty turn.

Pick the 3-5 strongest charts. Good candidates from the data we have:

- peak RSS reduction across the memopt waves (rss family)
- paged-vs-flash correctness by model across w4/w5 (pct family)
- findings by severity (donut), or findings by status (fixed/open/non-repro)
- throughput (tg/pp tps) where it did NOT regress - the "we saved memory
  without sacrificing speed" chart

State the narrative arc out loud (in your response, before rendering) so the
user can course-correct before the expensive render step.

## Phase 3 - Report (render the PDF)

Load the `pdf` skill and follow its `briefs/report.md` production workflow.
Key steps, in order:

1. `palette.generate --title "Tessera Research <date>" --mode minimal` - copy
   the output; every color below comes from it.
2. Render charts with matplotlib (Agg backend), saving each as PNG. Honor
   `typesetting/charts.md`: anti-overlap pre-check, generous padding, derive
   colors from the palette, use semantic colors (muted green/red) only for
   positive/negative deltas.
3. Assemble the ReportLab document:
   - Cover page (mandatory for a >=3 page report; see `typesetting/cover.md`)
   - TOC + bookmark navigation (so it is navigable, not just static)
   - Executive summary (1 paragraph: the headline result + the honesty turn)
   - One section per wave that moved the story, each with its chart + a tight
     table of the metrics that matter + prose tying it to the findings
   - Findings section: severity-grouped table; status column shows commit SHA
     for fixed items so a reviewer can verify
   - Open questions / non-reproductions (the honesty turn, in prose)
4. Preflight per the report brief: `code.sanitize` -> execute -> `font.check`
   -> `toc.check` -> `pages.clean` -> `pdf_qa.py`.
5. Output to `docs/reports/tessera-research-<YYYY-MM-DD>.pdf`. Create
   `docs/reports/` if it does not exist.

## Phase 4 - Video script (fill the template)

Read `templates/video-narrative.md` (next to this SKILL.md) and fill it in:

- Pull the thesis and arc from Phase 2.
- Map each scene to a chart you rendered in Phase 3, citing the report page.
- Write narration as it would be spoken (not as essay prose). Aim ~150 wpm.
- The honesty turn is a required scene, not optional.
- Save alongside the PDF: `docs/reports/tessera-research-<YYYY-MM-DD>-video.md`.

## Output contract

After running, you should have shipped:

```
docs/reports/tessera-research-<YYYY-MM-DD>.pdf         # navigable report
docs/reports/tessera-research-<YYYY-MM-DD>-video.md    # filled video script
/tmp/tessera-report/{metrics_long.csv,findings.csv}    # the data backing both
```

Plus an in-response summary to the user: the headline result, the honesty
turn, and pointers to the two deliverables.

## Failure handling

- If `scripts/alphaevolve-metrics.py` errors on missing polars, print the
  install hint from the script's own stderr and stop. Do not silently fall
  back to a partial report.
- If a wave's ledger is malformed, the script already warns and skips it;
  surface that in the report's caveats section rather than hiding it.
- If there are zero findings (fresh repo), say so explicitly - do not invent a
  findings narrative.
