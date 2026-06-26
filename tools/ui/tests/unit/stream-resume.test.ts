import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';

// node env unit project has no DOM, install a minimal localStorage backed by a Map
beforeAll(() => {
	const store = new Map<string, string>();
	const polyfill: Storage = {
		get length() {
			return store.size;
		},
		clear: () => store.clear(),
		getItem: (k) => (store.has(k) ? store.get(k)! : null),
		key: (i) => Array.from(store.keys())[i] ?? null,
		removeItem: (k) => {
			store.delete(k);
		},
		setItem: (k, v) => {
			store.set(k, String(v));
		}
	};
	(globalThis as unknown as { localStorage: Storage }).localStorage = polyfill;
});

import { ChatService } from '$lib/services/chat.service';
import { STREAM_RESUME_LOCALSTORAGE_KEY_PREFIX } from '$lib/constants';

describe('ChatService stream resume', () => {
	beforeEach(() => {
		localStorage.clear();
	});
	afterEach(() => {
		localStorage.clear();
	});

	it('returns null when no state exists for the conversation', () => {
		expect(ChatService.getStreamState('conv-a')).toBeNull();
	});

	it('saves and reads back the byte count', () => {
		ChatService.saveStreamState('conv-a', 4242);
		const got = ChatService.getStreamState('conv-a');
		expect(got).not.toBeNull();
		expect(got!.bytesReceived).toBe(4242);
		expect(typeof got!.updatedAt).toBe('number');
	});

	it('overwrites the previous byte count on a new save for the same conversation', () => {
		ChatService.saveStreamState('conv-a', 100);
		ChatService.saveStreamState('conv-a', 200);
		const got = ChatService.getStreamState('conv-a');
		expect(got!.bytesReceived).toBe(200);
	});

	it('keeps states for distinct conversations isolated', () => {
		ChatService.saveStreamState('conv-a', 10);
		ChatService.saveStreamState('conv-b', 20);
		expect(ChatService.getStreamState('conv-a')!.bytesReceived).toBe(10);
		expect(ChatService.getStreamState('conv-b')!.bytesReceived).toBe(20);
	});

	it('clears the state for a given conversation', () => {
		ChatService.saveStreamState('conv-a', 10);
		ChatService.clearStreamState('conv-a');
		expect(ChatService.getStreamState('conv-a')).toBeNull();
	});

	it('ignores empty conversation id on save', () => {
		ChatService.saveStreamState('', 1);
		expect(ChatService.getStreamState('')).toBeNull();
	});

	it('returns null on corrupted storage payload', () => {
		localStorage.setItem(`${STREAM_RESUME_LOCALSTORAGE_KEY_PREFIX}conv-a`, '{not-json');
		expect(ChatService.getStreamState('conv-a')).toBeNull();
	});

	it('persists the model alongside the byte count', () => {
		ChatService.saveStreamState('conv-a', 10, 'model-x');
		expect(ChatService.getStreamState('conv-a')!.model).toBe('model-x');
	});

	it('stores a null model when none is provided', () => {
		ChatService.saveStreamState('conv-a', 10);
		expect(ChatService.getStreamState('conv-a')!.model).toBeNull();
	});

	it('overwrites the model on a new save for the same conversation', () => {
		ChatService.saveStreamState('conv-a', 10, 'model-x');
		ChatService.saveStreamState('conv-a', 20, 'model-y');
		expect(ChatService.getStreamState('conv-a')!.model).toBe('model-y');
	});

	describe('resumeStreamIdentity', () => {
		it('appends the persisted model so the resume key matches the frozen POST identity', () => {
			ChatService.saveStreamState('conv-a', 10, 'model-x');
			expect(
				ChatService.resumeStreamIdentity('conv-a', ChatService.getStreamState('conv-a'), 'dropdown')
			).toBe('conv-a::model-x');
		});

		it('keeps the bare conv id when the persisted model is null', () => {
			ChatService.saveStreamState('conv-a', 10);
			expect(
				ChatService.resumeStreamIdentity('conv-a', ChatService.getStreamState('conv-a'), 'dropdown')
			).toBe('conv-a');
		});

		it('falls back to the current model only when no state is persisted', () => {
			expect(ChatService.resumeStreamIdentity('conv-a', null, 'dropdown')).toBe('conv-a::dropdown');
		});

		it('ignores the fallback when a state exists, the persisted value is authoritative', () => {
			ChatService.saveStreamState('conv-a', 10, 'model-x');
			expect(
				ChatService.resumeStreamIdentity('conv-a', ChatService.getStreamState('conv-a'), 'dropdown')
			).toBe('conv-a::model-x');
		});

		it('falls back when a legacy state has no model field', () => {
			localStorage.setItem(
				`${STREAM_RESUME_LOCALSTORAGE_KEY_PREFIX}conv-a`,
				JSON.stringify({ bytesReceived: 10, updatedAt: 1 })
			);
			expect(
				ChatService.resumeStreamIdentity('conv-a', ChatService.getStreamState('conv-a'), 'dropdown')
			).toBe('conv-a::dropdown');
		});
	});
});
