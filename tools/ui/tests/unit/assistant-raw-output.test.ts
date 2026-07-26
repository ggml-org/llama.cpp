import { describe, expect, it } from 'vitest';
import { AgenticSectionType } from '$lib/enums';
import { REASONING_TAGS } from '$lib/constants';
import { buildAssistantRawOutput, type AgenticSection } from '$lib/utils/agentic';

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
		const sections = [makeSection({ type: AgenticSectionType.REASONING, content: 'thinking...' })];
		expect(buildAssistantRawOutput(sections)).toBe(
			`${REASONING_TAGS.START}\nthinking...${REASONING_TAGS.END}`
		);
	});

	it('formats a text section as-is', () => {
		const sections = [makeSection({ type: AgenticSectionType.TEXT, content: 'Hello' })];
		expect(buildAssistantRawOutput(sections)).toBe('Hello');
	});

	it('formats a tool call with JSON args and no result label', () => {
		const sections = [
			makeSection({
				type: AgenticSectionType.TOOL_CALL,
				toolName: 'read_file',
				toolArgs: JSON.stringify({ path: '/tmp/file.txt' }),
				toolResult: 'file contents'
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
			makeSection({ type: AgenticSectionType.TEXT, content: 'Hello' }),
			makeSection({ type: AgenticSectionType.TOOL_CALL, toolName: 'noop' })
		];
		expect(buildAssistantRawOutput(sections)).toBe('Hello\n\n{\n  "name": "noop"\n}');
	});

	it('falls back to raw string args when JSON parsing fails', () => {
		const sections = [
			makeSection({
				type: AgenticSectionType.TOOL_CALL,
				toolName: 'broken',
				toolArgs: '{not json',
				toolResult: 'result'
			})
		];
		expect(buildAssistantRawOutput(sections)).toBe(
			['{', '  "name": "broken",', '  "arguments": "{not json"', '}', '', '', 'result'].join('\n')
		);
	});
});
