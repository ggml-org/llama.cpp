import { REASONING_TAGS } from '$lib/constants';
import { AgenticSectionType } from '$lib/enums';
import { type AgenticSection, buildAssistantRawOutput } from '$lib/utils/agentic';
import { describe, expect, it } from 'vitest';

function makeSection(
	overrides: Partial<AgenticSection> & { type: AgenticSectionType }
): AgenticSection {
	return {
		content: '',
		...overrides
	};
}

describe('buildAssistantRawOutput', () => {
	it('returns empty string for empty sections', () => {
		expect(buildAssistantRawOutput([])).toBe('');
	});

	it('formats a reasoning section with a single newline between tags and content', () => {
		const sections = [makeSection({ content: 'thinking...', type: AgenticSectionType.REASONING })];

		expect(buildAssistantRawOutput(sections)).toBe(
			`${REASONING_TAGS.START}\nthinking...${REASONING_TAGS.END}`
		);
	});

	it('formats a text section as-is', () => {
		const sections = [makeSection({ content: 'Hello', type: AgenticSectionType.TEXT })];

		expect(buildAssistantRawOutput(sections)).toBe('Hello');
	});

	it('formats a tool call with JSON args and no result label', () => {
		const sections = [
			makeSection({
				toolArgs: JSON.stringify({ path: '/tmp/file.txt' }),
				toolName: 'read_file',
				toolResult: 'file contents',
				type: AgenticSectionType.TOOL_CALL
			})
		];

		expect(buildAssistantRawOutput(sections)).toBe(
			[
				'{',
				'  "name": "read_file",',
				'  "arguments": {',
				'    "path": "/tmp/file.txt"',
				'  }',
				'}',
				'',
				'',
				'file contents'
			].join('\n')
		);
	});

	it('joins multiple sections with double newlines', () => {
		const sections = [
			makeSection({ content: 'Hello', type: AgenticSectionType.TEXT }),
			makeSection({ toolName: 'noop', type: AgenticSectionType.TOOL_CALL })
		];

		expect(buildAssistantRawOutput(sections)).toBe('Hello\n\n{\n  "name": "noop"\n}');
	});

	it('falls back to raw string args when JSON parsing fails', () => {
		const sections = [
			makeSection({
				toolArgs: '{not json',
				toolName: 'broken',
				toolResult: 'result',
				type: AgenticSectionType.TOOL_CALL
			})
		];

		expect(buildAssistantRawOutput(sections)).toBe(
			['{', '  "name": "broken",', '  "arguments": "{not json"', '}', '', '', 'result'].join('\n')
		);
	});
});
