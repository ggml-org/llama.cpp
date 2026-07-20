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

	it('settles on the basename once the path is complete', () => {
		const meta = parseReadFileMeta(section('read_file', argsBlob));
		expect(meta?.fileName).toBe('code.ts');
	});

	it('never shows a shrinking or non-monotonic header mid-stream', () => {
		const seen: string[] = [];

		for (const prefix of streamPrefixes(argsBlob)) {
			const meta = parseReadFileMeta(section('read_file', prefix));
			if (meta?.fileName) seen.push(meta.fileName);
		}

		// Each header must either extend the previous one or be the final
		// basename. Anything else is a visible flicker.
		const final = seen[seen.length - 1];
		const flickers = seen.filter(
			(value, i) => i > 0 && !value.startsWith(seen[i - 1]) && value !== final
		);

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
