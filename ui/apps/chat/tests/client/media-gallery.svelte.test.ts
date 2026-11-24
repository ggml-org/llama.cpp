// Guards the conversation slideshow: a thumbnail opens every image, audio clip
// and video in the open thread, in thread order, at the item that was clicked,
// whether it came from a user attachment or a tool result.

import { AttachmentType, MessageRole, MessageType } from '#lib/enums/index.ts';
import { conversationsStore, mediaGalleryStore } from '#lib/stores/index.ts';
import type { DatabaseMessage, DatabaseMessageExtra } from '#lib/types/index.ts';
import { afterEach, describe, expect, it } from 'vitest';

function image(name: string): DatabaseMessageExtra {
	return { base64Url: `data:image/png;base64,${name}`, name, type: AttachmentType.IMAGE };
}

function message(id: string, role: MessageRole, extra: DatabaseMessageExtra[]): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: 'conv-1',
		extra,
		id,
		parent: null,
		role,
		timestamp: 0,
		toolCalls: '',
		type: MessageType.TEXT
	};
}

afterEach(() => {
	conversationsStore.activeMessages = [];
	mediaGalleryStore.open = false;
});

describe('mediaGalleryStore', () => {
	it('opens the media of the whole thread at the clicked item', () => {
		const pasted = image('pasted');
		const shot = image('shot');
		const text: DatabaseMessageExtra = {
			content: 'x',
			name: 'notes.txt',
			type: AttachmentType.TEXT
		};

		conversationsStore.activeMessages = [
			message('u1', MessageRole.USER, [pasted, text]),
			message('t1', MessageRole.TOOL, [shot])
		];

		mediaGalleryStore.show(shot);

		expect(mediaGalleryStore.open).toBe(true);
		expect(mediaGalleryStore.items).toEqual([pasted, shot]);
		expect(mediaGalleryStore.index).toBe(1);
	});

	it('finds an item passed as a copy of the stored one', () => {
		conversationsStore.activeMessages = [message('t1', MessageRole.TOOL, [image('a'), image('b')])];

		mediaGalleryStore.show({ ...image('b') });

		expect(mediaGalleryStore.index).toBe(1);
		expect(mediaGalleryStore.items).toHaveLength(2);
	});

	it('opens an item outside the thread on its own', () => {
		conversationsStore.activeMessages = [message('t1', MessageRole.TOOL, [image('a')])];

		const stray = image('elsewhere');

		mediaGalleryStore.show(stray);

		expect(mediaGalleryStore.items).toEqual([stray]);
		expect(mediaGalleryStore.index).toBe(0);
	});
});
