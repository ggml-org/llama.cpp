# Tessera Research - Video Narrative Script

<!--
Tool-agnostic scaffold. Fill per scene; feed to any narrator/video generator,
or read it yourself for a self-record. Every claim must trace to a metric in
the accompanying PDF report (cite the page) or a findings entry (cite the ref).
No fabricated numbers, no hand-waving.
-->

**Title:** <working title>
**Audience:** <engineers / researchers / general technical>
**Target length:** <minutes>
**Source report:** docs/reports/tessera-research-<date>.pdf
**One-line thesis:** <the single takeaway a viewer should leave with>

## Narrative arc (fill before scene-by-scene)

- **Hook** (the single most surprising result, stated in one sentence):
- **Problem** (what was broken / suboptimal, and why it matters):
- **Approach** (one phrase per wave that contributed):
- **Honesty turn** (what did NOT work or is still open - required, not optional):
- **Outro / CTA** (what the viewer should do next):

## Scene-by-scene

For each scene: duration, what the viewer SEES (visual / chart / B-roll), what
they HEAR (narration, written out as you'd speak it), and the chart/report ref.
Aim for 6-10 scenes total.

| # | Duration | Visual | Narration | Ref |
|---|----------|--------|-----------|-----|
| 1 | 0:00-0:15 | <cover frame / title card> | <hook line, spoken> | report p.1 |
| 2 | 0:15-0:45 | <problem framing - code snippet / diagram> | <set up the problem> | findings <ref> |
| 3 | 0:45-1:30 | <chart: headline metric across waves> | <walk the wave-over-wave change> | report p.<n>, metric <key> |
| 4 | 1:30-2:15 | <chart: findings by severity, or correctness delta> | <what the data shows> | report p.<n> |
| 5 | 2:15-2:45 | <honesty turn: open questions / non-reproductions> | <what we do NOT claim> | findings <ref>, status open |
| 6 | 2:45-3:00 | <outro frame> | <CTA> | - |

## B-roll / asset checklist

Charts to pre-render (matplotlib, palette from report) so they're ready to drop in:

- [ ] <chart 1: e.g. peak-RSS reduction across waves>
- [ ] <chart 2: e.g. paged-vs-flash correctness climbing 1% -> 100%>
- [ ] <chart 3: e.g. findings by severity donut>

## Notes for the narrator

- Pace for ~150 words/minute; keep technical terms but define each on first use.
- The honesty turn (scene 5) is non-negotiable - cutting it undermines the rest.
- If a number is in the narration, it MUST appear on screen at the same time.
