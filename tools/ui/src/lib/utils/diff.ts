/**
 * Line-based unified diff helper.
 *
 * Produces output in the unified-diff text format that highlight.js's
 * built-in `diff` language understands (`hljs-addition` / `hljs-deletion`
 * spans). Reused by CodeDiff; kept dependency-free to avoid adding a
 * third-party diff library.
 */

export interface DiffHunk {
	oldStart: number;
	newStart: number;
	lines: string[];
}

/**
 * Computes a unified diff between two strings at line granularity using
 * an LCS dynamic-programming table. Returns hunks with `+`/`-`/` ` prefixed
 * lines, identical in spirit to `git diff` (minus the file headers).
 */
export function computeLineDiff(oldText: string, newText: string): DiffHunk[] {
	const oldLines = oldText.length === 0 ? [] : oldText.split('\n');
	const newLines = newText.length === 0 ? [] : newText.split('\n');

	// LCS length table (dp[i+1][j+1] = lcs of oldLines[0..i] and newLines[0..j])
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

	// Walk the table to recover the edit script as { type, line } records
	type Record = { type: 'context' | 'add' | 'del'; line: string };
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

	// Group records into hunks with a shared context window so the diff
	// stays readable without dumping the whole file in one block.
	const CONTEXT = 3;
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
		// Trim trailing context beyond the window
		let end = pending.length;
		let extra = 0;
		while (end > 0 && pending[end - 1].startsWith(' ')) {
			extra++;
			if (extra > CONTEXT) {
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
			// Start a fresh hunk window if we've drifted past the context limit
			if (hasChange && contextRun >= CONTEXT) {
				flush();
				hunkOldStart = oldLine;
				hunkNewStart = newLine;
			}
			pending.push(' ' + r.line);
			contextRun++;
			oldLine++;
			newLine++;
		} else if (r.type === 'del') {
			pending.push('-' + r.line);
			hasChange = true;
			contextRun = 0;
			oldLine++;
		} else {
			pending.push('+' + r.line);
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
			const oldLen = h.lines.filter((l) => l.startsWith('-') && !l.startsWith('---')).length;
			const newLen = h.lines.filter((l) => l.startsWith('+') && !l.startsWith('+++')).length;
			const header = `@@ -${h.oldStart},${oldLen} +${h.newStart},${newLen} @@`;
			return header + '\n' + h.lines.join('\n');
		})
		.join('\n');
}

/** Convenience: true when the two texts differ after trimming. */
export function hasContentDiff(oldText: string, newText: string): boolean {
	return oldText.trim() !== newText.trim();
}
