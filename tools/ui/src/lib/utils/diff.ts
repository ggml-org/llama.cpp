/**
 * Line-based unified diff helper.
 *
 * Produces output in the unified-diff text format that highlight.js's
 * `diff` language understands (`hljs-addition` / `hljs-deletion`
 * spans). Reused by CodeDiff; kept dependency-free to avoid adding a
 * third-party diff library.
 */

import { DIFF_CONTEXT_WINDOW, DIFF_LINE_PREFIX, DIFF_HEADER_MARKER } from '$lib/constants';

export interface DiffHunk {
	oldStart: number;
	newStart: number;
	lines: string[];
}

type Record = { type: 'context' | 'add' | 'del'; line: string };

/**
 * Computes a unified diff between two strings at line granularity using
 * an LCS dynamic-programming table. Returns hunks with `+`/`-`/` ` prefixed
 * lines, identical in spirit to `git diff` (minus the file headers).
 */
export function computeLineDiff(oldText: string, newText: string): DiffHunk[] {
	const oldLines = oldText.length === 0 ? [] : oldText.split('\n');
	const newLines = newText.length === 0 ? [] : newText.split('\n');

	// dp[i+1][j+1] = LCS length of oldLines[0..i] and newLines[0..j]
	const dp: number[][] = Array.from({ length: oldLines.length + 1 }, () =>
		new Array(newLines.length + 1).fill(0)
	);

	for (let i = oldLines.length - 1; i >= 0; i--) {
		for (let j = newLines.length - 1; j >= 0; j--) {
			if (oldLines[i] === newLines[j]) {
				dp[i][j] = dp[i + 1][j + 1] + 1;
			} else {
				dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
			}
		}
	}

	const records: Record[] = [];
	let i = 0;
	let j = 0;
	while (i < oldLines.length && j < newLines.length) {
		if (oldLines[i] === newLines[j]) {
			records.push({ type: 'context', line: oldLines[i] });
			i++;
			j++;
		} else if (dp[i + 1][j] >= dp[i][j + 1]) {
			records.push({ type: 'del', line: oldLines[i] });
			i++;
		} else {
			records.push({ type: 'add', line: newLines[j] });
			j++;
		}
	}
	while (i < oldLines.length) records.push({ type: 'del', line: oldLines[i++] });
	while (j < newLines.length) records.push({ type: 'add', line: newLines[j++] });

	const hunks: DiffHunk[] = [];
	let oldLine = 1;
	let newLine = 1;
	let pending: string[] = [];
	let hunkOldStart = 1;
	let hunkNewStart = 1;
	let hasChange = false;
	let contextRun = 0;

	function flush() {
		if (!hasChange) {
			pending = [];
			return;
		}
		let end = pending.length;
		let extra = 0;
		while (end > 0 && pending[end - 1].startsWith(DIFF_LINE_PREFIX.CONTEXT)) {
			extra++;
			if (extra > DIFF_CONTEXT_WINDOW) {
				pending.pop();
				end--;
			} else {
				break;
			}
		}
		hunks.push({
			oldStart: hunkOldStart,
			newStart: hunkNewStart,
			lines: pending.slice()
		});
		pending = [];
		hasChange = false;
		contextRun = 0;
	}

	for (const r of records) {
		if (r.type === 'context') {
			if (hasChange && contextRun >= DIFF_CONTEXT_WINDOW) {
				flush();
				hunkOldStart = oldLine;
				hunkNewStart = newLine;
			}
			pending.push(DIFF_LINE_PREFIX.CONTEXT + r.line);
			contextRun++;
			oldLine++;
			newLine++;
		} else if (r.type === 'del') {
			pending.push(DIFF_LINE_PREFIX.DEL + r.line);
			hasChange = true;
			contextRun = 0;
			oldLine++;
		} else {
			pending.push(DIFF_LINE_PREFIX.ADD + r.line);
			hasChange = true;
			contextRun = 0;
			newLine++;
		}
	}
	flush();

	return hunks;
}

/**
 * Renders the computed hunks as a single unified-diff text block suitable
 * for highlight.js's `diff` language.
 */
export function formatUnifiedDiff(hunks: DiffHunk[]): string {
	return hunks
		.map((h) => {
			const oldLen = h.lines.filter(
				(l) => l.startsWith(DIFF_LINE_PREFIX.DEL) && !l.startsWith(DIFF_HEADER_MARKER.DEL)
			).length;
			const newLen = h.lines.filter(
				(l) => l.startsWith(DIFF_LINE_PREFIX.ADD) && !l.startsWith(DIFF_HEADER_MARKER.ADD)
			).length;
			const header = `${DIFF_HEADER_MARKER.HUNK} -${h.oldStart},${oldLen} +${h.newStart},${newLen} ${DIFF_HEADER_MARKER.HUNK}`;
			return header + '\n' + h.lines.join('\n');
		})
		.join('\n');
}

/** Convenience: true when the two texts differ after trimming. */
export function hasContentDiff(oldText: string, newText: string): boolean {
	return oldText.trim() !== newText.trim();
}
