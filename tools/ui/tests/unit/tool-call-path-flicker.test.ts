// Reproduces the mid-stream flicker in read_file / write_file headers.
//
// Tool args arrive token by token, so `path` is a partial string for most of the
// call. Deriving a basename from it makes the header walk through every path
// segment ("Us" -> "za" -> "Dev" -> ... -> "foo.ts"), and deriving a language
// from it makes the syntax highlighter swap grammars mid-render.

import { describe, expect, it } from 'vitest';
import { parseReadFileMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/read-file';
import { parseWriteFileMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/write-file';
import { AgenticSectionType } from '../../src/lib/enums';
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
