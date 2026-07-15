/**
 * Line-level unified diff for tool result rendering.
 *
 * Pure functions: no DOM, no Svelte, no highlight.js dependency. The
 * returned `DiffLine[]` carries enough information both to render a
 * custom diff block (per-entry kind/text) and to fold back into a
 * unified-diff-format string for off-the-shelf highlighter languages
 * (`renderUnifiedDiff`).
 *
 * Algorithm: LCS dynamic programming with a soft "remove before add"
 * tiebreak so the resulting diff reads `(old -> new)` left to right.
 * O(m*n) time/space which is fine for the handful of lines an
 * `edit_file` snippet typically carries.
 */

import { DiffLineKind } from '$lib/enums';

export interface DiffLine {
	kind: DiffLineKind;
	text: string;
	/** 1-indexed line number in the OLD content. Undefined for `add` lines. */
	oldLine?: number;
	/** 1-indexed line number in the NEW content. Undefined for `remove` lines. */
	newLine?: number;
}

export function computeLineDiff(oldText: string, newText: string): DiffLine[] {
	const oldLines = splitLines(oldText);
	const newLines = splitLines(newText);

	const m = oldLines.length;
	const n = newLines.length;

	if (m === 0 && n === 0) return [];
	if (m === 0) return newLines.map((t, k) => ({ kind: DiffLineKind.ADD, text: t, newLine: k + 1 }));
	if (n === 0)
		return oldLines.map((t, k) => ({ kind: DiffLineKind.REMOVE, text: t, oldLine: k + 1 }));

	const lcs: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
	for (let i = 1; i <= m; i++) {
		for (let j = 1; j <= n; j++) {
			if (oldLines[i - 1] === newLines[j - 1]) {
				lcs[i][j] = lcs[i - 1][j - 1] + 1;
			} else {
				lcs[i][j] = Math.max(lcs[i - 1][j], lcs[i][j - 1]);
			}
		}
	}

	const result: DiffLine[] = [];
	let i = m;
	let j = n;
	while (i > 0 && j > 0) {
		if (oldLines[i - 1] === newLines[j - 1]) {
			result.push({
				kind: DiffLineKind.CONTEXT,
				text: oldLines[i - 1],
				oldLine: i,
				newLine: j
			});
			i--;
			j--;
		} else if (lcs[i - 1][j] >= lcs[i][j - 1]) {
			result.push({ kind: DiffLineKind.REMOVE, text: oldLines[i - 1], oldLine: i });
			i--;
		} else {
			result.push({ kind: DiffLineKind.ADD, text: newLines[j - 1], newLine: j });
			j--;
		}
	}
	while (i > 0) {
		result.push({ kind: DiffLineKind.REMOVE, text: oldLines[i - 1], oldLine: i });
		i--;
	}
	while (j > 0) {
		result.push({ kind: DiffLineKind.ADD, text: newLines[j - 1], newLine: j });
		j--;
	}

	result.reverse();
	return result;
}

/** Folds `DiffLine[]` into a unified-diff-format text (`` ` ``/`+`/`-` prefixes).
 *  Pass to a diff-aware highlighter (e.g., SyntaxHighlightedCode with
 *  `language="diff"`) for colorization.
 */
export function renderUnifiedDiff(lines: DiffLine[]): string {
	if (lines.length === 0) return '';
	return lines.map((l) => prefixFor(l.kind) + l.text).join('\n');
}

/** Column-1 marker for a `DiffLine`: ` `, `+`, or `-`. */
export function prefixFor(kind: DiffLineKind): string {
	if (kind === DiffLineKind.ADD) return '+';
	if (kind === DiffLineKind.REMOVE) return '-';
	return ' ';
}

function splitLines(text: string): string[] {
	if (text === '') return [];
	const parts = text.split('\n');
	if (parts[parts.length - 1] === '') parts.pop();
	return parts.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));
}
