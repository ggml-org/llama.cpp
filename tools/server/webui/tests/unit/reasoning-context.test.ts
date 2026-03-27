import { describe, it, expect } from 'vitest';
import { AGENTIC_REGEX, REASONING_TAGS } from '$lib/constants/agentic';
import { ContentPartType } from '$lib/enums';

// Replicate ChatService.extractReasoningFromContent (private static)
function extractReasoningFromContent(
	content: string | Array<{ type: string; text?: string }> | null | undefined
): string | undefined {
	if (!content) return undefined;

	const extractFromString = (text: string): string => {
		const parts: string[] = [];
		const re = new RegExp(AGENTIC_REGEX.REASONING_EXTRACT.source);
		let match = re.exec(text);
		while (match) {
			parts.push(match[1]);
			text = text.slice(match.index + match[0].length);
			match = re.exec(text);
		}
		return parts.join('');
	};

	if (typeof content === 'string') {
		const result = extractFromString(content);
		return result || undefined;
	}

	if (!Array.isArray(content)) return undefined;

	const parts: string[] = [];
	for (const part of content) {
		if (part.type === ContentPartType.TEXT && part.text) {
			const result = extractFromString(part.text);
			if (result) parts.push(result);
		}
	}
	return parts.length > 0 ? parts.join('') : undefined;
}

// Replicate ChatService.stripReasoningContent (private static)
function stripReasoningContent(
	content: string | Array<{ type: string; text?: string }> | null | undefined
): typeof content {
	if (!content) return content;

	if (typeof content === 'string') {
		return content
			.replace(AGENTIC_REGEX.REASONING_BLOCK, '')
			.replace(AGENTIC_REGEX.REASONING_OPEN, '')
			.replace(AGENTIC_REGEX.AGENTIC_TOOL_CALL_BLOCK, '')
			.replace(AGENTIC_REGEX.AGENTIC_TOOL_CALL_OPEN, '');
	}

	if (!Array.isArray(content)) return content;

	return content.map((part) => {
		if (part.type !== ContentPartType.TEXT || !part.text) return part;
		return {
			...part,
			text: part.text
				.replace(AGENTIC_REGEX.REASONING_BLOCK, '')
				.replace(AGENTIC_REGEX.REASONING_OPEN, '')
				.replace(AGENTIC_REGEX.AGENTIC_TOOL_CALL_BLOCK, '')
				.replace(AGENTIC_REGEX.AGENTIC_TOOL_CALL_OPEN, '')
		};
	});
}

// Simulate the message mapping logic from ChatService.sendMessage
function buildApiMessage(
	content: string,
	excludeReasoningFromContext: boolean
): { role: string; content: string; reasoning_content?: string } {
	const cleaned = stripReasoningContent(content) as string;
	const mapped: { role: string; content: string; reasoning_content?: string } = {
		role: 'assistant',
		content: cleaned
	};
	if (!excludeReasoningFromContext) {
		const reasoning = extractReasoningFromContent(content);
		if (reasoning) {
			mapped.reasoning_content = reasoning;
		}
	}
	return mapped;
}

// Helper: wrap reasoning the same way the chat store does during streaming
function wrapReasoning(reasoning: string, content: string): string {
	return `${REASONING_TAGS.START}${reasoning}${REASONING_TAGS.END}${content}`;
}

describe('reasoning content extraction', () => {
	it('extracts reasoning from tagged string content', () => {
		const input = wrapReasoning('step 1, step 2', 'The answer is 42.');
		const result = extractReasoningFromContent(input);
		expect(result).toBe('step 1, step 2');
	});

	it('returns undefined when no reasoning tags present', () => {
		expect(extractReasoningFromContent('Just a normal response.')).toBeUndefined();
	});

	it('returns undefined for null/empty input', () => {
		expect(extractReasoningFromContent(null)).toBeUndefined();
		expect(extractReasoningFromContent(undefined)).toBeUndefined();
		expect(extractReasoningFromContent('')).toBeUndefined();
	});

	it('extracts reasoning from content part arrays', () => {
		const input = [
			{
				type: ContentPartType.TEXT,
				text: wrapReasoning('thinking hard', 'result')
			}
		];
		expect(extractReasoningFromContent(input)).toBe('thinking hard');
	});

	it('handles multiple reasoning blocks', () => {
		const input =
			REASONING_TAGS.START +
			'block1' +
			REASONING_TAGS.END +
			'middle' +
			REASONING_TAGS.START +
			'block2' +
			REASONING_TAGS.END +
			'end';
		expect(extractReasoningFromContent(input)).toBe('block1block2');
	});

	it('ignores non-text content parts', () => {
		const input = [{ type: 'image_url', text: wrapReasoning('hidden', 'img') }];
		expect(extractReasoningFromContent(input)).toBeUndefined();
	});
});

describe('strip reasoning content', () => {
	it('removes reasoning tags from string content', () => {
		const input = wrapReasoning('internal thoughts', 'visible answer');
		expect(stripReasoningContent(input)).toBe('visible answer');
	});

	it('removes reasoning from content part arrays', () => {
		const input = [
			{
				type: ContentPartType.TEXT,
				text: wrapReasoning('thoughts', 'answer')
			}
		];
		const result = stripReasoningContent(input) as Array<{ type: string; text?: string }>;
		expect(result[0].text).toBe('answer');
	});
});

describe('API message building with reasoning preservation', () => {
	const storedContent = wrapReasoning('Let me think: 2+2=4, basic arithmetic.', 'The answer is 4.');

	it('preserves reasoning_content when excludeReasoningFromContext is false', () => {
		const msg = buildApiMessage(storedContent, false);
		expect(msg.content).toBe('The answer is 4.');
		expect(msg.reasoning_content).toBe('Let me think: 2+2=4, basic arithmetic.');
		// no internal tags leak into either field
		expect(msg.content).not.toContain('<<<');
		expect(msg.reasoning_content).not.toContain('<<<');
	});

	it('strips reasoning_content when excludeReasoningFromContext is true', () => {
		const msg = buildApiMessage(storedContent, true);
		expect(msg.content).toBe('The answer is 4.');
		expect(msg.reasoning_content).toBeUndefined();
	});

	it('handles content with no reasoning in both modes', () => {
		const plain = 'No reasoning here.';
		const msgPreserve = buildApiMessage(plain, false);
		const msgExclude = buildApiMessage(plain, true);
		expect(msgPreserve.content).toBe(plain);
		expect(msgPreserve.reasoning_content).toBeUndefined();
		expect(msgExclude.content).toBe(plain);
		expect(msgExclude.reasoning_content).toBeUndefined();
	});

	it('cleans agentic tool call blocks from content even when preserving reasoning', () => {
		const input =
			wrapReasoning('plan', 'text') +
			'\n\n<<<AGENTIC_TOOL_CALL_START>>>\n' +
			'<<<TOOL_NAME:bash>>>\n' +
			'<<<TOOL_ARGS_START>>>\n{}\n<<<TOOL_ARGS_END>>>\nout\n' +
			'<<<AGENTIC_TOOL_CALL_END>>>\n';
		const msg = buildApiMessage(input, false);
		expect(msg.content).not.toContain('<<<');
		expect(msg.reasoning_content).toBe('plan');
	});
});
