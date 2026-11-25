import { buildEvictedImageNote, READ_MEDIA_ATTACHMENT_REGEX } from '#lib/constants/index.ts';
import { AttachmentType, MessageRole, MessageType } from '#lib/enums/index.ts';
import type { ApiChatMessageData } from '#lib/types/api.js';
import type { DatabaseMessage, DatabaseMessageExtra } from '#lib/types/index.ts';
import { evictOldImages, findReferencedImage } from '#lib/utils/index.ts';
import { describe, expect, it } from 'vitest';

const KEEP_TURNS = 2;
const IMAGE_URL = 'data:image/png;base64,AAAA';

function image(name: string): DatabaseMessageExtra {
	return { base64Url: IMAGE_URL, name, type: AttachmentType.IMAGE };
}

function makeMessage(
	id: string,
	role: MessageRole,
	content: string,
	extra?: DatabaseMessageExtra[]
): DatabaseMessage {
	return {
		children: [],
		content,
		convId: 'conv',
		extra,
		id,
		parent: null,
		role,
		timestamp: 0,
		type: MessageType.TEXT
	} as DatabaseMessage;
}

// `count` turns, each a user message and its answer; the first user message holds an image
function turns(count: number): DatabaseMessage[] {
	return Array.from({ length: count }, (_, i) => [
		makeMessage(
			`user-${i}-id`,
			MessageRole.USER,
			`question ${i}`,
			i === 0 ? [image('cat.png')] : undefined
		),
		makeMessage(`assistant-${i}-id`, MessageRole.ASSISTANT, `answer ${i}`)
	]).flat();
}

describe('evictOldImages', () => {
	it('returns the list as it is while no turn is far behind', () => {
		const messages = turns(2 * KEEP_TURNS - 1);

		expect(evictOldImages(messages, KEEP_TURNS)).toBe(messages);
	});

	it('puts a note in the place of an image of an old turn', () => {
		const messages = turns(2 * KEEP_TURNS);
		const [first, ...rest] = evictOldImages(messages, KEEP_TURNS);

		expect(first.extra).toEqual([]);
		expect(first.content).toBe(
			`question 0\n${buildEvictedImageNote('cat.png', 'attachment:user-0-i/1')}`
		);
		expect(rest).toEqual(messages.slice(1));
		// the stored message keeps its image
		expect(messages[0].extra).toEqual([image('cat.png')]);
	});

	it('moves the cut once per keepTurns turns, not every turn', () => {
		const withImageAt = (count: number, at: number) => {
			const messages = turns(count);

			messages[at * 2].extra = [image('late.png')];

			return evictOldImages(messages, KEEP_TURNS)[at * 2].extra;
		};

		// turn 2 is 2 turns behind, then 3, and still has its image
		expect(withImageAt(4, 2)).toHaveLength(1);
		expect(withImageAt(5, 2)).toHaveLength(1);
		// the cut moved past it
		expect(withImageAt(6, 2)).toHaveLength(0);
	});

	it('keeps the attachments that are no images', () => {
		const text: DatabaseMessageExtra = {
			content: 'notes',
			name: 'notes.txt',
			type: AttachmentType.TEXT
		};
		const messages = turns(2 * KEEP_TURNS);

		messages[0].extra = [text, image('cat.png')];

		const [first] = evictOldImages(messages, KEEP_TURNS);

		expect(first.extra).toEqual([text]);
		expect(first.content).toContain('attachment:user-0-i/2');
	});

	it('writes the note alone when the message had no text', () => {
		const messages = turns(2 * KEEP_TURNS);

		messages[0].content = '';

		expect(evictOldImages(messages, KEEP_TURNS)[0].content).toBe(
			buildEvictedImageNote('cat.png', 'attachment:user-0-i/1')
		);
	});

	it('does not count a compaction node as a turn', () => {
		const node = {
			...makeMessage('c1', MessageRole.USER, 'summary'),
			type: MessageType.COMPACTION
		} as DatabaseMessage;
		const messages = [node, ...turns(2 * KEEP_TURNS - 1)];

		expect(evictOldImages(messages, KEEP_TURNS)).toBe(messages);
	});

	it('leaves messages that are already in the API shape alone', () => {
		const messages: ApiChatMessageData[] = Array.from({ length: 2 * KEEP_TURNS }, () => ({
			content: 'hi',
			role: MessageRole.USER
		}));

		expect(evictOldImages(messages, KEEP_TURNS)).toEqual(messages);
	});
});

describe('findReferencedImage', () => {
	const messages = [
		makeMessage('user-0-id', MessageRole.USER, 'look', [
			{ content: 'notes', name: 'notes.txt', type: AttachmentType.TEXT },
			image('cat.png')
		])
	];

	it('finds the image a note points to', () => {
		const [first] = evictOldImages([...messages, ...turns(2 * KEEP_TURNS)], KEEP_TURNS);
		const reference = first.content.match(/path "([^"]+)"/)?.[1] ?? '';

		expect(reference).toMatch(READ_MEDIA_ATTACHMENT_REGEX);
		expect(findReferencedImage(reference, messages)).toEqual(image('cat.png'));
	});

	it('finds nothing for an attachment that is no image', () => {
		expect(findReferencedImage('attachment:user-0-i/1', messages)).toBeUndefined();
	});

	it('finds nothing for an unknown message, position or shape', () => {
		expect(findReferencedImage('attachment:other-id/2', messages)).toBeUndefined();
		expect(findReferencedImage('attachment:user-0-i/3', messages)).toBeUndefined();
		expect(findReferencedImage('attachment:cat.png', messages)).toBeUndefined();
	});
});
