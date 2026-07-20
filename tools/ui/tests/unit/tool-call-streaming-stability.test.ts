// Streaming-stability contract for every tool-call meta parser.
//
// ADDING A TOOL BLOCK? Add its parser here. The check replays the args blob one
// character at a time and fails if the block would ever show a value that later
// changes into something unrelated - which is what produces visible flicker.
//
// See helpers/streaming-stability.ts for the two invariants and why they exist.

import { describe, expect, it } from 'vitest';
import { AgenticSectionType, BuiltInTool } from '$lib/enums';
import { DEFAULT_LANGUAGE } from '$lib/constants';
import type { AgenticSection } from '$lib/utils';
import { findStreamingViolations } from './helpers/streaming-stability';

import { parseReadFileMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/read-file';
import { parseWriteFileMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/write-file';
import { parseEditFileMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/edit-file';
import { parseExecShellCommandMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/exec-shell-command';
import { parseGrepSearchMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/grep-search';
import { parseFileGlobSearchMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/file-glob-search';
import { parseRunJavascriptMeta } from '../../src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/run-javascript';

const FILE_PATH = '/Users/dev/project/src/lib/utils/code.ts';

function sectionFor(toolName: string, argsBlob: string): AgenticSection {
	return {
		type: AgenticSectionType.TOOL_CALL_STREAMING,
		content: '',
		toolName,
		toolArgs: argsBlob,
		toolCallId: 'call_1'
	};
}

interface ParserCase {
	name: string;
	tool: BuiltInTool;
	parse: (section: AgenticSection) => unknown;
	args: unknown;
	placeholders?: Record<string, unknown[]>;
}

const CASES: ParserCase[] = [
	{
		name: 'read_file (path only)',
		tool: BuiltInTool.READ_FILE,
		parse: parseReadFileMeta,
		args: { path: FILE_PATH },
		// The highlight language cannot be known until the extension has arrived.
		placeholders: { language: [DEFAULT_LANGUAGE] }
	},
	{
		name: 'read_file (with line range)',
		tool: BuiltInTool.READ_FILE,
		parse: parseReadFileMeta,
		args: { path: FILE_PATH, start_line: 12, end_line: 480 },
		placeholders: { language: [DEFAULT_LANGUAGE], lineRange: [null] }
	},
	{
		name: 'write_file',
		tool: BuiltInTool.WRITE_FILE,
		parse: parseWriteFileMeta,
		args: { path: FILE_PATH, content: 'const a = 1;\nconst b = 2;\nexport { a, b };\n' },
		placeholders: { language: [DEFAULT_LANGUAGE] }
	},
	{
		name: 'edit_file (single edit)',
		tool: BuiltInTool.EDIT_FILE,
		parse: parseEditFileMeta,
		args: {
			path: FILE_PATH,
			edits: [{ old_text: 'const a = 1;\nconst b = 2;', new_text: 'const a = 10;\nconst b = 2;' }]
		}
	},
	{
		name: 'edit_file (multiple edits)',
		tool: BuiltInTool.EDIT_FILE,
		parse: parseEditFileMeta,
		args: {
			path: FILE_PATH,
			edits: [
				{ old_text: 'alpha', new_text: 'ALPHA' },
				{ old_text: 'beta\ngamma', new_text: 'BETA\nGAMMA' }
			]
		}
	},
	{
		name: 'exec_shell_command',
		tool: BuiltInTool.EXEC_SHELL_COMMAND,
		parse: parseExecShellCommandMeta,
		args: { command: 'grep -rn "needle" src/ | head -50' }
	},
	{
		name: 'grep_search',
		tool: BuiltInTool.GREP_SEARCH,
		parse: parseGrepSearchMeta,
		args: { pattern: 'TODO\\(perf\\)', path: 'src/lib', include: '*.ts' }
	},
	{
		name: 'file_glob_search',
		tool: BuiltInTool.FILE_GLOB_SEARCH,
		parse: parseFileGlobSearchMeta,
		// This parser reads `path` + `include`; the stability helper flagged a
		// `pattern`-only fixture as an invalid case rather than a parser bug.
		args: { path: 'src/lib', include: '**/*.svelte' }
	},
	{
		name: 'run_javascript',
		tool: BuiltInTool.RUN_JAVASCRIPT,
		parse: parseRunJavascriptMeta,
		args: { code: 'const total = [1, 2, 3].reduce((a, b) => a + b, 0);\nconsole.log(total);' }
	}
];

describe('tool-call parsers are stable while their args stream in', () => {
	for (const testCase of CASES) {
		it(`${testCase.name}: never shows a value it later changes`, () => {
			const violations = findStreamingViolations({
				parse: (argsBlob) => testCase.parse(sectionFor(testCase.tool, argsBlob)),
				args: testCase.args,
				placeholders: testCase.placeholders
			});

			expect(violations).toEqual([]);
		});
	}

	it('covers every parser that has a tool block', () => {
		// Guards against a new tool block being added without a case above.
		const covered = new Set(CASES.map((testCase) => testCase.tool));

		expect([...covered].sort()).toEqual(
			[
				BuiltInTool.READ_FILE,
				BuiltInTool.WRITE_FILE,
				BuiltInTool.EDIT_FILE,
				BuiltInTool.EXEC_SHELL_COMMAND,
				BuiltInTool.GREP_SEARCH,
				BuiltInTool.FILE_GLOB_SEARCH,
				BuiltInTool.RUN_JAVASCRIPT
			].sort()
		);
	});
});
