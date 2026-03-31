import { describe, it, expect } from 'vitest';
import { AGENTIC_REGEX } from '$lib/constants/agentic';

// Mirror the logic in ChatService.stripReasoningContent so we can test it in isolation.
// The real function is private static, so we replicate the strip pipeline here.
function stripContextMarkers(content: string): string {
	return content
		.replace(AGENTIC_REGEX.REASONING_BLOCK, '')
		.replace(AGENTIC_REGEX.REASONING_OPEN, '')
		.replace(AGENTIC_REGEX.AGENTIC_TOOL_CALL_BLOCK, '')
		.replace(AGENTIC_REGEX.AGENTIC_TOOL_CALL_OPEN, '');
}

// A realistic complete tool call block as stored in message.content after a turn.
const COMPLETE_BLOCK =
	'\n\n<<<AGENTIC_TOOL_CALL_START>>>\n' +
	'<<<TOOL_NAME:bash_tool>>>\n' +
	'<<<TOOL_ARGS_START>>>\n' +
	'{"command":"ls /tmp","description":"list tmp"}\n' +
	'<<<TOOL_ARGS_END>>>\n' +
	'file1.txt\nfile2.txt\n' +
	'<<<AGENTIC_TOOL_CALL_END>>>\n';

// Partial block: streaming was cut before END arrived.
const OPEN_BLOCK =
	'\n\n<<<AGENTIC_TOOL_CALL_START>>>\n' +
	'<<<TOOL_NAME:bash_tool>>>\n' +
	'<<<TOOL_ARGS_START>>>\n' +
	'{"command":"ls /tmp","description":"list tmp"}\n' +
	'<<<TOOL_ARGS_END>>>\n' +
	'partial output...';

describe('agentic marker stripping for context', () => {
	it('strips a complete tool call block, leaving surrounding text', () => {
		const input = 'Before.' + COMPLETE_BLOCK + 'After.';
		const result = stripContextMarkers(input);
		// markers gone; residual newlines between fragments are fine
		expect(result).not.toContain('<<<');
		expect(result).toContain('Before.');
		expect(result).toContain('After.');
	});

	it('strips multiple complete tool call blocks', () => {
		const input = 'A' + COMPLETE_BLOCK + 'B' + COMPLETE_BLOCK + 'C';
		const result = stripContextMarkers(input);
		expect(result).not.toContain('<<<');
		expect(result).toContain('A');
		expect(result).toContain('B');
		expect(result).toContain('C');
	});

	it('strips an open/partial tool call block (no END marker)', () => {
		const input = 'Lead text.' + OPEN_BLOCK;
		const result = stripContextMarkers(input);
		expect(result).toBe('Lead text.');
		expect(result).not.toContain('<<<');
	});

	it('does not alter content with no markers', () => {
		const input = 'Just a normal assistant response.';
		expect(stripContextMarkers(input)).toBe(input);
	});

	it('strips reasoning block independently', () => {
		const input = '<<<reasoning_content_start>>>think hard<<<reasoning_content_end>>>Answer.';
		expect(stripContextMarkers(input)).toBe('Answer.');
	});

	it('strips both reasoning and agentic blocks together', () => {
		const input =
			'<<<reasoning_content_start>>>plan<<<reasoning_content_end>>>' +
			'Some text.' +
			COMPLETE_BLOCK;
		expect(stripContextMarkers(input)).not.toContain('<<<');
		expect(stripContextMarkers(input)).toContain('Some text.');
	});

	it('empty string survives', () => {
		expect(stripContextMarkers('')).toBe('');
	});
});
