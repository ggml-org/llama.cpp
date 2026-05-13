import { describe, it, expect } from 'vitest';
import { ChatService } from '$lib/services/chat.service';
import { AttachmentType, ContentPartType, MessageRole } from '$lib/enums';
import type { ApiChatMessageContentPart, DatabaseMessage, DatabaseMessageExtra } from '$lib/types';

/**
 * Regression for ggml-org/llama.cpp#22962: when the user pastes a long blob
 * into the chat input it is uploaded as a "Pasted" text attachment. The
 * resulting API message must place the attachment text BEFORE the user's
 * typed instruction so that the typed query stays at the suffix and the
 * pasted/uploaded data forms a stable prefix for cache reuse across
 * different questions over the same data.
 */

function makeBaseMessage(
	content: string,
	extra: DatabaseMessageExtra[]
): DatabaseMessage & { extra: DatabaseMessageExtra[] } {
	return {
		id: 'm1',
		convId: 'c1',
		timestamp: 0,
		role: MessageRole.USER,
		content,
		parent: null,
		children: [],
		extra
	} as unknown as DatabaseMessage & { extra: DatabaseMessageExtra[] };
}

function expectText(part: ApiChatMessageContentPart): string {
	expect(part.type).toBe(ContentPartType.TEXT);
	expect(part.text).toBeDefined();
	return part.text as string;
}

describe('ChatService.convertDbMessageToApiChatMessageData attachment ordering', () => {
	it('emits pasted text attachment before the typed instruction', () => {
		const message = makeBaseMessage('Summarize in 1 sentence', [
			{
				type: AttachmentType.TEXT,
				name: 'Pasted',
				content: 'LONG_PASTED_BLOB'
			}
		]);

		const result = ChatService.convertDbMessageToApiChatMessageData(message);
		const parts = result.content as ApiChatMessageContentPart[];

		expect(Array.isArray(parts)).toBe(true);
		expect(parts).toHaveLength(2);

		const [first, second] = parts;
		expect(expectText(first)).toContain('LONG_PASTED_BLOB');
		expect(expectText(second)).toBe('Summarize in 1 sentence');
	});

	it('places the typed instruction after every attachment kind', () => {
		const message = makeBaseMessage('What does this say?', [
			{ type: AttachmentType.IMAGE, name: 'pic.png', base64Url: 'data:image/png;base64,AAA' },
			{ type: AttachmentType.TEXT, name: 'notes.txt', content: 'TEXT_FILE_CONTENT' },
			{ type: AttachmentType.LEGACY_CONTEXT, name: 'old.txt', content: 'LEGACY_CONTENT' }
		]);

		const parts = ChatService.convertDbMessageToApiChatMessageData(message)
			.content as ApiChatMessageContentPart[];

		const lastPart = parts[parts.length - 1];
		expect(expectText(lastPart)).toBe('What does this say?');

		const typedIndex = parts.length - 1;
		const textAttachmentIndex = parts.findIndex(
			(p) => p.type === ContentPartType.TEXT && p.text?.includes('TEXT_FILE_CONTENT')
		);
		const legacyIndex = parts.findIndex(
			(p) => p.type === ContentPartType.TEXT && p.text?.includes('LEGACY_CONTENT')
		);
		const imageIndex = parts.findIndex((p) => p.type === ContentPartType.IMAGE_URL);

		expect(textAttachmentIndex).toBeGreaterThanOrEqual(0);
		expect(legacyIndex).toBeGreaterThanOrEqual(0);
		expect(imageIndex).toBeGreaterThanOrEqual(0);
		expect(textAttachmentIndex).toBeLessThan(typedIndex);
		expect(legacyIndex).toBeLessThan(typedIndex);
		expect(imageIndex).toBeLessThan(typedIndex);
	});

	it('omits the typed-text part when the user only pasted content', () => {
		const message = makeBaseMessage('', [
			{ type: AttachmentType.TEXT, name: 'Pasted', content: 'ONLY_BLOB' }
		]);

		const parts = ChatService.convertDbMessageToApiChatMessageData(message)
			.content as ApiChatMessageContentPart[];

		expect(parts).toHaveLength(1);
		expect(expectText(parts[0])).toContain('ONLY_BLOB');
	});
});
