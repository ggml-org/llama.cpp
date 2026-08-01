import { describe, expect, it } from 'vitest';
import { AgenticSectionType, BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/utils';
import { parseToolArgs } from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/_shared';
import {
	parseWriteFileMeta,
	type WriteFileMeta
} from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/write-file';
import { parseEditFileMeta } from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/edit-file';
import { parseReadFileMeta } from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/read-file';
import { parseGrepSearchMeta } from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/grep-search';
import { parseFileGlobSearchMeta } from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/file-glob-search';
import { parseRunJavascriptMeta } from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/run-javascript';
import { parseExecShellCommandMeta } from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/exec-shell-command';

function makeSection(
	overrides: Partial<AgenticSection> = {},
	toolName = BuiltInTool.READ_FILE
): AgenticSection {
	return {
		type: AgenticSectionType.TOOL_CALL,
		content: '',
		toolName,
		toolArgs: JSON.stringify({ path: '/foo.txt' }),
		toolResult: undefined,
		...overrides
	};
}

describe('parseToolArgs (shared)', () => {
	it('returns null when the section has no toolArgs', () => {
		const result = parseToolArgs(BuiltInTool.READ_FILE, makeSection({ toolArgs: undefined }));
		expect(result).toBeNull();
	});

	it('returns null when the tool name does not match', () => {
		const result = parseToolArgs(
			BuiltInTool.READ_FILE,
			makeSection({ toolArgs: '{"path":"/x"}' }, BuiltInTool.WRITE_FILE)
		);
		expect(result).toBeNull();
	});

	it('returns null when args are not valid final JSON (partial: false)', () => {
		const result = parseToolArgs(
			BuiltInTool.READ_FILE,
			makeSection({ toolArgs: '{"path": "/foo.tx' })
		);
		expect(result).toBeNull();
	});

	it('returns parsed args when valid final JSON', () => {
		const result = parseToolArgs(
			BuiltInTool.READ_FILE,
			makeSection({ toolArgs: '{"path":"/foo.txt"}' })
		);
		expect(result).toEqual({ path: '/foo.txt' });
	});

	it('accepts partial JSON when partial: true', () => {
		const result = parseToolArgs(
			BuiltInTool.READ_FILE,
			makeSection({ toolArgs: '{"path": "/foo.tx' }),
			{ partial: true }
		);
		expect(result).toEqual({ path: '/foo.tx' });
	});
});

describe('parseWriteFileMeta', () => {
	it('returns null for sections with a different tool name', () => {
		expect(
			parseWriteFileMeta(
				makeSection({ toolName: BuiltInTool.READ_FILE, toolArgs: '{"path":"/x","content":"y"}' })
			)
		).toBeNull();
	});

	it('returns null when args have no path-like field', () => {
		expect(
			parseWriteFileMeta(
				makeSection({ toolName: BuiltInTool.WRITE_FILE, toolArgs: '{"content":"x"}' })
			)
		).toBeNull();
	});

	it('accepts partial args (renders incrementally as content streams in)', () => {
		const meta = parseWriteFileMeta(
			makeSection({ toolName: BuiltInTool.WRITE_FILE, toolArgs: '{"path":"/foo.t' })
		);
		expect(meta?.filePath).toBe('/foo.t');
	});

	it('returns file path, language, content, bytes, resultMessage', () => {
		const meta = parseWriteFileMeta(
			makeSection(
				{
					toolName: BuiltInTool.WRITE_FILE,
					toolArgs: '{"path":"/foo.ts","content":"x"}',
					toolResult: '{"result":"wrote","bytes":42}'
				},
				BuiltInTool.WRITE_FILE
			)
		);
		expect(meta).toMatchObject<Partial<WriteFileMeta>>({
			filePath: '/foo.ts',
			language: expect.any(String),
			content: 'x',
			bytesWritten: 42,
			resultMessage: 'wrote'
		});
	});

	it('surfaces errorMessage from the result blob', () => {
		const meta = parseWriteFileMeta(
			makeSection({
				toolName: BuiltInTool.WRITE_FILE,
				toolArgs: '{"path":"/foo","content":"x"}',
				toolResult: '{"error":"permission denied"}'
			})
		);
		expect(meta?.errorMessage).toBe('permission denied');
	});
});

describe('parseEditFileMeta', () => {
	it('parses edits array and applies editsApplied from the result', () => {
		const section = makeSection(
			{
				toolName: BuiltInTool.EDIT_FILE,
				toolArgs:
					'{"path":"/foo.ts","edits":[{"old_text":"a","new_text":"b"},{"old_text":"c","new_text":"d"}]}',
				toolResult: '{"result":"ok","edits_applied":2}'
			},
			BuiltInTool.EDIT_FILE
		);
		const meta = parseEditFileMeta(section);
		expect(meta?.edits).toEqual([
			{ oldText: 'a', newText: 'b' },
			{ oldText: 'c', newText: 'd' }
		]);
		expect(meta?.editsApplied).toBe(2);
		expect(meta?.resultMessage).toBe('ok');
	});

	it('drops edits with empty old_text', () => {
		const section = makeSection(
			{
				toolName: BuiltInTool.EDIT_FILE,
				toolArgs: '{"path":"/foo","edits":[{"old_text":""},{"old_text":"a","new_text":""}]}'
			},
			BuiltInTool.EDIT_FILE
		);
		const meta = parseEditFileMeta(section);
		// First entry is dropped (empty old_text). Second is kept
		// (empty new_text is fine - it's the "delete" case).
		expect(meta?.edits).toEqual([{ oldText: 'a', newText: '' }]);
	});

	it('errorMessage wins over result message', () => {
		const section = makeSection(
			{
				toolName: BuiltInTool.EDIT_FILE,
				toolArgs: '{"path":"/foo"}',
				toolResult: '{"error":"bad path","result":"ok"}'
			},
			BuiltInTool.EDIT_FILE
		);
		const meta = parseEditFileMeta(section);
		expect(meta?.errorMessage).toBe('bad path');
		expect(meta?.resultMessage).toBeUndefined();
	});
});

describe('parseReadFileMeta', () => {
	it('parses file name alone (no range)', () => {
		const meta = parseReadFileMeta(
			makeSection({ toolArgs: '{"path":"/foo.txt"}' }, BuiltInTool.READ_FILE)
		);
		expect(meta?.fileName).toBe('foo.txt');
		expect(meta?.lineRange).toBeNull();
	});

	it('parses start_line + end_line into a range', () => {
		const meta = parseReadFileMeta(
			makeSection(
				{ toolArgs: '{"path":"/foo.ts","start_line":10,"end_line":20}' },
				BuiltInTool.READ_FILE
			)
		);
		expect(meta?.lineRange).toEqual({ start: 10, end: 20 });
	});

	it('parses start_line + line_count into a range', () => {
		const meta = parseReadFileMeta(
			makeSection(
				{ toolArgs: '{"path":"/foo.ts","start_line":10,"line_count":5}' },
				BuiltInTool.READ_FILE
			)
		);
		expect(meta?.lineRange).toEqual({ start: 10, end: 14 });
	});

	it('returns null when args cannot be parsed', () => {
		expect(parseReadFileMeta(makeSection({ toolArgs: '{bad' }, BuiltInTool.READ_FILE))).toBeNull();
	});
});

describe('parseGrepSearchMeta', () => {
	it('returns null when path or pattern is missing', () => {
		expect(
			parseGrepSearchMeta(
				makeSection({ toolName: BuiltInTool.GREP_SEARCH, toolArgs: '{"pattern":"foo"}' })
			)
		).toBeNull();
		expect(
			parseGrepSearchMeta(
				makeSection({ toolName: BuiltInTool.GREP_SEARCH, toolArgs: '{"path":"/x"}' })
			)
		).toBeNull();
	});

	it('parses structured plain_text_response into matches', () => {
		const meta = parseGrepSearchMeta(
			makeSection(
				{
					toolName: BuiltInTool.GREP_SEARCH,
					toolArgs: '{"path":"/x","pattern":"foo"}',
					toolResult: JSON.stringify({ plain_text_response: 'a.ts:hello\nb.ts:world' })
				},
				BuiltInTool.GREP_SEARCH
			)
		);
		expect(meta?.matches).toHaveLength(2);
		expect(meta?.matches[0]).toEqual({ file: 'a.ts', content: 'hello' });
	});

	it('falls back to raw-text parsing when result is not JSON', () => {
		const meta = parseGrepSearchMeta(
			makeSection(
				{
					toolName: BuiltInTool.GREP_SEARCH,
					toolArgs: '{"path":"/x","pattern":"foo"}',
					toolResult: 'a.ts:hello\nb.ts:world'
				},
				BuiltInTool.GREP_SEARCH
			)
		);
		expect(meta?.matches).toHaveLength(2);
	});

	it('parses line numbers when return_line_numbers is true', () => {
		const meta = parseGrepSearchMeta(
			makeSection(
				{
					toolName: BuiltInTool.GREP_SEARCH,
					toolArgs: '{"path":"/x","pattern":"foo","return_line_numbers":true}',
					toolResult: 'a.ts:12:hello'
				},
				BuiltInTool.GREP_SEARCH
			)
		);
		expect(meta?.matches[0]).toEqual({ file: 'a.ts', line: 12, content: 'hello' });
		expect(meta?.showLineNumbers).toBe(true);
	});
});

describe('parseFileGlobSearchMeta', () => {
	it('falls back to raw-text parsing when result is not JSON', () => {
		const meta = parseFileGlobSearchMeta(
			makeSection(
				{
					toolName: BuiltInTool.FILE_GLOB_SEARCH,
					toolArgs: '{"path":"/x"}',
					toolResult: 'a.ts\nb.ts'
				},
				BuiltInTool.FILE_GLOB_SEARCH
			)
		);
		expect(meta?.matches).toEqual(['a.ts', 'b.ts']);
	});

	it('parses plain_text_response from a JSON object', () => {
		const meta = parseFileGlobSearchMeta(
			makeSection(
				{
					toolName: BuiltInTool.FILE_GLOB_SEARCH,
					toolArgs: '{"path":"/x"}',
					toolResult: JSON.stringify({ plain_text_response: 'a.ts\nb.ts' })
				},
				BuiltInTool.FILE_GLOB_SEARCH
			)
		);
		expect(meta?.matches).toEqual(['a.ts', 'b.ts']);
	});

	it('surfaces errorMessage from the result blob', () => {
		const meta = parseFileGlobSearchMeta(
			makeSection(
				{
					toolName: BuiltInTool.FILE_GLOB_SEARCH,
					toolArgs: '{"path":"/x"}',
					toolResult: JSON.stringify({ error: 'permission denied' })
				},
				BuiltInTool.FILE_GLOB_SEARCH
			)
		);
		expect(meta?.errorMessage).toBe('permission denied');
	});
});

describe('parseRunJavascriptMeta', () => {
	it('returns null when code is missing', () => {
		expect(
			parseRunJavascriptMeta(makeSection({ toolName: BuiltInTool.RUN_JAVASCRIPT, toolArgs: '{}' }))
		).toBeNull();
	});

	it('reads code and timeout', () => {
		const meta = parseRunJavascriptMeta(
			makeSection(
				{ toolName: BuiltInTool.RUN_JAVASCRIPT, toolArgs: '{"code":"Math.PI","timeout_ms":5000}' },
				BuiltInTool.RUN_JAVASCRIPT
			)
		);
		expect(meta?.code).toBe('Math.PI');
		expect(meta?.timeoutMs).toBe(5000);
	});

	it('reads error field from a JSON-object result', () => {
		const meta = parseRunJavascriptMeta(
			makeSection(
				{
					toolName: BuiltInTool.RUN_JAVASCRIPT,
					toolArgs: '{"code":"throw new Error()"}',
					toolResult: JSON.stringify({ error: 'undefined is not a function' })
				},
				BuiltInTool.RUN_JAVASCRIPT
			)
		);
		expect(meta?.errorMessage).toBe('undefined is not a function');
	});

	it('does NOT treat a JSON-array result as an error', () => {
		// SandboxService returns successful output as a JSON array;
		// only JSON objects carry `error`. Raw arrays must round-trip
		// through unchanged.
		const meta = parseRunJavascriptMeta(
			makeSection(
				{
					toolName: BuiltInTool.RUN_JAVASCRIPT,
					toolArgs: '{"code":"[1,2,3]"}',
					toolResult: '[1,2,3]'
				},
				BuiltInTool.RUN_JAVASCRIPT
			)
		);
		expect(meta?.errorMessage).toBeUndefined();
	});

	it('scans a non-JSON string result for an `Error:` line', () => {
		const meta = parseRunJavascriptMeta(
			makeSection(
				{
					toolName: BuiltInTool.RUN_JAVASCRIPT,
					toolArgs: '{"code":"foo"}',
					toolResult: 'Error: undefined is not a function\n  at <anonymous>:1:1'
				},
				BuiltInTool.RUN_JAVASCRIPT
			)
		);
		expect(meta?.errorMessage).toBe('undefined is not a function');
	});
});

describe('parseExecShellCommandMeta', () => {
	it('reads command from the args', () => {
		const meta = parseExecShellCommandMeta(
			makeSection(
				{ toolName: BuiltInTool.EXEC_SHELL_COMMAND, toolArgs: '{"command":"ls -la"}' },
				BuiltInTool.EXEC_SHELL_COMMAND
			)
		);
		expect(meta?.command).toBe('ls -la');
	});

	it('accepts cmd / shell_command aliases', () => {
		expect(
			parseExecShellCommandMeta(
				makeSection(
					{ toolName: BuiltInTool.EXEC_SHELL_COMMAND, toolArgs: '{"cmd":"ls"}' },
					BuiltInTool.EXEC_SHELL_COMMAND
				)
			)?.command
		).toBe('ls');
		expect(
			parseExecShellCommandMeta(
				makeSection(
					{ toolName: BuiltInTool.EXEC_SHELL_COMMAND, toolArgs: '{"shell_command":"ls"}' },
					BuiltInTool.EXEC_SHELL_COMMAND
				)
			)?.command
		).toBe('ls');
	});

	it('returns null when no command alias is present', () => {
		expect(
			parseExecShellCommandMeta(
				makeSection(
					{ toolName: BuiltInTool.EXEC_SHELL_COMMAND, toolArgs: '{"cwd":"/x"}' },
					BuiltInTool.EXEC_SHELL_COMMAND
				)
			)
		).toBeNull();
	});
});
