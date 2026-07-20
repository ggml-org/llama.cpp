// Reproduces the mid-stream flicker in read_file / write_file headers.
//
// Tool args arrive token by token, so `path` is a partial string for most of the
// call. Deriving a basename from it makes the header walk through every path
// segment ("Us" -> "za" -> "Dev" -> ... -> "foo.ts"), and deriving a language
// from it makes the syntax highlighter swap grammars mid-render.

import { describe, expect, it } from 'vitest';
import { parseReadFileMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/read-file';
import { parseWriteFileMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/write-file';
import { parseEditFileMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/edit-file';
import { computeLineDiff } from '../../src/lib/utils/compute-line-diff';
import { AgenticSectionType, DiffLineKind } from '../../src/lib/enums';
import type { AgenticSection } from '../../src/lib/utils';

const FULL_PATH = '/Users/zach/Dev/llama.cpp/tools/ui/src/lib/utils/code.ts';

function section(toolName: string, argsBlob: string): AgenticSection {
	return {
		type: AgenticSectionType.TOOL_CALL_STREAMING,
		content: '',
		toolName,
		toolArgs: argsBlob,
		toolCallId: 'call_1'
	};
}

/** Every prefix of the args blob, as the stream would deliver them. */
function streamPrefixes(argsBlob: string): string[] {
	const out: string[] = [];
	for (let i = 1; i <= argsBlob.length; i++) out.push(argsBlob.slice(0, i));
	return out;
}

describe('read_file header stability while streaming', () => {
	const argsBlob = JSON.stringify({ path: FULL_PATH });

	it('shows the full path, matching write_file and edit_file', () => {
		const meta = parseReadFileMeta(section('read_file', argsBlob));
		expect(meta?.filePath).toBe(FULL_PATH);
	});

	it('never shows a shrinking or non-monotonic header mid-stream', () => {
		const seen: string[] = [];

		for (const prefix of streamPrefixes(argsBlob)) {
			const meta = parseReadFileMeta(section('read_file', prefix));
			if (meta?.filePath) seen.push(meta.filePath);
		}

		// Rendering the raw path makes this strictly monotonic - every frame
		// extends the previous one, with no transition at the end.
		const flickers = seen.filter((value, i) => i > 0 && !value.startsWith(seen[i - 1]));

		expect(flickers).toEqual([]);
	});

	it('does not swap the highlight language while the path streams', () => {
		const languages = new Set<string>();

		for (const prefix of streamPrefixes(argsBlob)) {
			const meta = parseReadFileMeta(section('read_file', prefix));
			if (meta) languages.add(meta.language);
		}

		// A language may go from "unknown" to the real one exactly once; more than
		// two distinct values means the highlighter is thrashing.
		expect(languages.size).toBeLessThanOrEqual(2);
	});
});

describe('multi-param tool calls do not blank out between parameters', () => {
	// While a *key name* streams (`,"start_l`), naive closure yields
	// `{"path":"...","start_l"}` - invalid JSON. That made the parser return null
	// for every such frame, so the whole block rendered empty between parameters.
	const argsBlob = JSON.stringify({ path: FULL_PATH, start_line: 1, end_line: 40 });

	it('keeps the path visible for every frame after it arrives', () => {
		const prefixes = streamPrefixes(argsBlob);
		const firstWithPath = prefixes.findIndex(
			(p) => parseReadFileMeta(section('read_file', p))?.filePath
		);

		expect(firstWithPath).toBeGreaterThan(-1);

		const blankFrames = prefixes
			.slice(firstWithPath)
			.map((p, i) => ({ i: i + firstWithPath, meta: parseReadFileMeta(section('read_file', p)) }))
			.filter((row) => !row.meta?.filePath);

		expect(blankFrames.map((row) => row.i)).toEqual([]);
	});

	it('still reports the final line range', () => {
		const meta = parseReadFileMeta(section('read_file', argsBlob));

		expect(meta?.lineRange).toEqual({ start: 1, end: 40 });
		expect(meta?.filePath).toBe(FULL_PATH);
	});

	it('never shows a partially-streamed line range', () => {
		// A range that appears then changes is the same class of flicker.
		const ranges = new Set<string>();

		for (const prefix of streamPrefixes(argsBlob)) {
			const meta = parseReadFileMeta(section('read_file', prefix));
			if (meta?.lineRange) ranges.add(`${meta.lineRange.start}-${meta.lineRange.end}`);
		}

		// Only the settled range should ever be displayed.
		expect([...ranges]).toEqual(['1-40']);
	});
});

describe('write_file header stability while streaming', () => {
	const argsBlob = JSON.stringify({ path: FULL_PATH, content: 'const a = 1;\nconst b = 2;\n' });

	it('never shows a shrinking or non-monotonic header mid-stream', () => {
		const seen: string[] = [];

		for (const prefix of streamPrefixes(argsBlob)) {
			const meta = parseWriteFileMeta(section('write_file', prefix));
			if (meta?.filePath) seen.push(meta.filePath);
		}

		const final = seen[seen.length - 1];
		const flickers = seen.filter(
			(value, i) => i > 0 && !value.startsWith(seen[i - 1]) && value !== final
		);

		expect(flickers).toEqual([]);
	});

	it('does not swap the highlight language while the path streams', () => {
		const languages = new Set<string>();

		for (const prefix of streamPrefixes(argsBlob)) {
			const meta = parseWriteFileMeta(section('write_file', prefix));
			if (meta) languages.add(meta.language);
		}

		expect(languages.size).toBeLessThanOrEqual(2);
	});
});

describe('edit_file diff stability while streaming', () => {
	// `old_text` streams to completion before `new_text` begins. In that window the
	// edit has a full old value and an empty new one, so a diff built from it shows
	// every line as deleted - then snaps back as the replacement arrives.
	const OLD = 'const a = 1;\nconst b = 2;\nconst c = 3;';
	const NEW = 'const a = 10;\nconst b = 2;\nconst c = 30;';
	const argsBlob = JSON.stringify({
		path: '/src/thing.ts',
		edits: [{ old_text: OLD, new_text: NEW }]
	});

	it('never renders an all-deletions diff mid-stream', () => {
		const oldLineCount = OLD.split('\n').length;
		const offenders: Array<{ at: number; removed: number; added: number }> = [];

		for (let i = 1; i <= argsBlob.length; i++) {
			const meta = parseEditFileMeta(section('edit_file', argsBlob.slice(0, i)));
			if (!meta || meta.edits.length === 0) continue;

			for (const edit of meta.edits) {
				const diff = computeLineDiff(edit.oldText, edit.newText);
				const removed = diff.filter((line) => line.kind === DiffLineKind.REMOVE).length;
				const added = diff.filter((line) => line.kind === DiffLineKind.ADD).length;

				// Every old line deleted and nothing added is the "snap" state.
				if (removed >= oldLineCount && added === 0) {
					offenders.push({ at: i, removed, added });
				}
			}
		}

		expect(offenders).toEqual([]);
	});

	it('still produces the correct final diff', () => {
		const meta = parseEditFileMeta(section('edit_file', argsBlob));

		expect(meta?.edits).toHaveLength(1);

		const diff = computeLineDiff(meta!.edits[0].oldText, meta!.edits[0].newText);
		const added = diff.filter((line) => line.kind === DiffLineKind.ADD).length;
		const removed = diff.filter((line) => line.kind === DiffLineKind.REMOVE).length;

		expect(added).toBe(2);
		expect(removed).toBe(2);
	});
});

describe('edit_file withholding does not swallow legitimate edits', () => {
	it('renders a genuine deletion once the args are complete', () => {
		const argsBlob = JSON.stringify({
			path: '/src/thing.ts',
			edits: [{ old_text: 'gone line one\ngone line two', new_text: '' }]
		});
		const meta = parseEditFileMeta(section('edit_file', argsBlob));

		expect(meta?.edits).toHaveLength(1);

		const diff = computeLineDiff(meta!.edits[0].oldText, meta!.edits[0].newText);
		const removed = diff.filter((line) => line.kind === DiffLineKind.REMOVE).length;

		expect(removed).toBe(2);
		expect(diff.filter((line) => line.kind === DiffLineKind.ADD)).toHaveLength(0);
	});

	it('keeps completed edits visible while a later one streams', () => {
		const full = JSON.stringify({
			path: '/src/thing.ts',
			edits: [
				{ old_text: 'aaa', new_text: 'AAA' },
				{ old_text: 'bbb', new_text: 'BBB' }
			]
		});

		// Cut partway through the second edit's new_text.
		const cut = full.indexOf('BBB');
		const meta = parseEditFileMeta(section('edit_file', full.slice(0, cut)));

		// The first edit is complete and must still render.
		expect(meta?.edits.length).toBeGreaterThanOrEqual(1);
		expect(meta?.edits[0]).toEqual({ oldText: 'aaa', newText: 'AAA' });
	});

	it('renders every edit once the call completes', () => {
		const full = JSON.stringify({
			path: '/src/thing.ts',
			edits: [
				{ old_text: 'aaa', new_text: 'AAA' },
				{ old_text: 'bbb', new_text: 'BBB' }
			]
		});
		const meta = parseEditFileMeta(section('edit_file', full));

		expect(meta?.edits).toEqual([
			{ oldText: 'aaa', newText: 'AAA' },
			{ oldText: 'bbb', newText: 'BBB' }
		]);
	});
});
