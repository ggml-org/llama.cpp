import { AttachmentType, MessageRole, MessageType } from '#lib/enums/index.ts';
import type { DatabaseMessage } from '#lib/types/database.js';
import { findMessageMatch, matchSnippet, searchableTexts } from '#lib/utils/conversation-search.ts';
import { describe, expect, it } from 'vitest';

function message(fields: Partial<DatabaseMessage>): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: 'conv',
		id: 'msg',
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		type: MessageType.TEXT,
		...fields
	} as DatabaseMessage;
}

describe('searchableTexts', () => {
	it('reads reasoning, content, tool calls and attachments, but no ids or payloads', () => {
		const texts = searchableTexts(
			message({
				content: 'the answer',
				extra: [
					{ content: 'notes inside', name: 'notes.txt', type: AttachmentType.TEXT },
					{ base64Url: 'data:image/png;base64,QUJD', name: 'cat.png', type: AttachmentType.IMAGE }
				] as DatabaseMessage['extra'],
				reasoningContent: 'thinking it over',
				toolCalls: JSON.stringify([
					{ function: { arguments: '{"q":"weather"}', name: 'web_search' }, id: 'call_secret' }
				])
			})
		);

		expect(texts).toEqual([
			'thinking it over',
			'the answer',
			'web_search',
			'{"q":"weather"}',
			'notes.txt',
			'notes inside',
			'cat.png'
		]);
		expect(texts.join(' ')).not.toContain('call_secret');
		expect(texts.join(' ')).not.toContain('QUJD');
	});

	it('skips synthetic messages and tolerates broken tool call JSON', () => {
		expect(searchableTexts(message({ content: 'cwd changed', isSynthetic: true }))).toEqual([]);
		expect(searchableTexts(message({ content: 'ok', toolCalls: '{not json' }))).toEqual(['ok']);
	});
});

describe('findMessageMatch', () => {
	it('matches case-insensitively and reports where', () => {
		expect(findMessageMatch(message({ content: 'Plan a trip to Paris' }), 'paris')).toEqual({
			at: 15,
			text: 'Plan a trip to Paris'
		});
	});

	it('finds tool output and reasoning', () => {
		const tool = message({ content: 'sunny, 21C', role: MessageRole.TOOL });
		const reasoning = message({ reasoningContent: 'user wants Lisbon' });

		expect(findMessageMatch(tool, 'sunny')).not.toBeNull();
		expect(findMessageMatch(reasoning, 'lisbon')).not.toBeNull();
		expect(findMessageMatch(tool, 'rain')).toBeNull();
	});
});

describe('matchSnippet', () => {
	it('keeps short text whole', () => {
		expect(matchSnippet('find me here', 5, 2)).toEqual({
			after: ' here',
			before: 'find ',
			match: 'me'
		});
	});

	it('cuts long text around the match on one line', () => {
		const text = `${'a'.repeat(100)}\n\nneedle\n${'b'.repeat(200)}`;
		const snippet = matchSnippet(text, 102, 6, 40);

		expect(snippet.match).toBe('needle');
		expect(snippet.before.startsWith('...')).toBe(true);
		expect(snippet.after.endsWith('...')).toBe(true);
		expect(snippet.before + snippet.match + snippet.after).not.toContain('\n');
	});
});
