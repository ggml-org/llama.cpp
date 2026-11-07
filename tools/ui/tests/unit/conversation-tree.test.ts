// Pure fork-tree and message-tree rules that every storage backend relies on.

import {
	CONFIG_LOCALSTORAGE_KEY,
	SIDEBAR_WIDTH_LOCALSTORAGE_KEY,
	STREAM_RESUME_LOCALSTORAGE_KEY_PREFIX
} from '$lib/constants';
import { MessageRole, MessageType, PreferenceScope } from '$lib/enums';
import { PreferencesService } from '$lib/services/preferences.service';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';
import {
	buildForkedConversation,
	collectForkDescendantIds,
	groupMessagesByConversation,
	reparentDirectForks
} from '$lib/utils/conversation-tree';
import { describe, expect, it } from 'vitest';

function conv(id: string, forkedFrom?: string): DatabaseConversation {
	return { currNode: '', forkedFromConversationId: forkedFrom, id, lastModified: 0, name: id };
}

function msg(
	id: string,
	parent: string | null,
	children: string[] = [],
	timestamp = 0
): DatabaseMessage {
	return {
		children,
		content: id,
		convId: 'c1',
		extra: [],
		id,
		parent,
		role: MessageRole.USER,
		timestamp,
		type: MessageType.TEXT
	} as DatabaseMessage;
}

describe('collectForkDescendantIds', () => {
	it('collects forks at every depth, not the root itself', () => {
		const convs = [conv('a'), conv('b', 'a'), conv('c', 'b'), conv('d', 'a'), conv('x')];

		expect(collectForkDescendantIds(convs, 'a').sort()).toEqual(['b', 'c', 'd']);
	});
});

describe('reparentDirectForks', () => {
	it('moves direct forks to the parent of the deleted conversation', () => {
		const convs = [conv('a'), conv('b', 'a'), conv('c', 'b')];

		expect(reparentDirectForks(convs, 'b')).toEqual([conv('c', 'a')]);
	});

	it('skips ancestors deleted in the same batch and drops forks being deleted', () => {
		const convs = [conv('a'), conv('b', 'a'), conv('c', 'b'), conv('d', 'c'), conv('e', 'c')];

		expect(reparentDirectForks(convs, 'c', new Set(['b', 'c', 'e']))).toEqual([conv('d', 'a')]);
	});

	it('promotes forks to top-level when the ancestry loops', () => {
		const convs = [conv('a', 'b'), conv('b', 'a'), conv('c', 'a')];

		expect(reparentDirectForks(convs, 'a', new Set(['a', 'b']))).toEqual([conv('c', undefined)]);
	});

	it('returns nothing for an unknown conversation', () => {
		expect(reparentDirectForks([conv('a')], 'missing')).toEqual([]);
	});
});

describe('buildForkedConversation', () => {
	// root -> u1 -> a1, with a sibling branch u1 -> a2
	const messages = [
		msg('root', null, ['u1'], 0),
		msg('u1', 'root', ['a1', 'a2'], 1),
		msg('a1', 'u1', [], 2),
		msg('a2', 'u1', [], 3)
	];

	it('clones the path to the target with fresh, consistently remapped ids', () => {
		const { conv: fork, messages: cloned } = buildForkedConversation(conv('src'), messages, 'a1', {
			includeAttachments: true,
			name: 'Fork'
		});

		expect(fork.forkedFromConversationId).toBe('src');
		expect(fork.name).toBe('Fork');
		expect(cloned.map((m) => m.content)).toEqual(['root', 'u1', 'a1']);
		expect(cloned.every((m) => m.convId === fork.id)).toBe(true);
		expect(fork.currNode).toBe(cloned[2].id);

		const [root, u1, a1] = cloned;

		expect(root.parent).toBeNull();
		expect(root.children).toEqual([u1.id]);
		expect(u1.parent).toBe(root.id);
		expect(u1.children).toEqual([a1.id]);
		expect(new Set(cloned.map((m) => m.id)).size).toBe(3);
		expect(cloned.some((m) => messages.some((orig) => orig.id === m.id))).toBe(false);
		expect(a1.extra).toEqual([]);
	});

	it('drops attachments when asked to', () => {
		const { messages: cloned } = buildForkedConversation(conv('src'), messages, 'a1', {
			includeAttachments: false,
			name: 'Fork'
		});

		expect(cloned.every((m) => m.extra === undefined)).toBe(true);
	});

	it('throws when there are no messages to fork', () => {
		expect(() =>
			buildForkedConversation(conv('src'), [], 'a1', { includeAttachments: true, name: 'Fork' })
		).toThrow();
	});
});

describe('groupMessagesByConversation', () => {
	it('groups by conversation, oldest first', () => {
		const groups = groupMessagesByConversation([
			{ ...msg('b2', null, [], 2), convId: 'b' },
			{ ...msg('a2', null, [], 2), convId: 'a' },
			{ ...msg('b1', null, [], 1), convId: 'b' }
		]);

		expect(groups.get('a')?.map((m) => m.id)).toEqual(['a2']);
		expect(groups.get('b')?.map((m) => m.id)).toEqual(['b1', 'b2']);
	});
});

describe('PreferencesService.scopeOf', () => {
	it('classifies user data and device-local keys', () => {
		expect(PreferencesService.scopeOf(CONFIG_LOCALSTORAGE_KEY)).toBe(PreferenceScope.USER);
		expect(PreferencesService.scopeOf(SIDEBAR_WIDTH_LOCALSTORAGE_KEY)).toBe(PreferenceScope.DEVICE);
		expect(PreferencesService.scopeOf(`${STREAM_RESUME_LOCALSTORAGE_KEY_PREFIX}conv-a`)).toBe(
			PreferenceScope.DEVICE
		);
	});
});
