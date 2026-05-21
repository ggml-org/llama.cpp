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

import {
	saveStreamState,
	getStreamState,
	clearStreamState,
	resumeStreamIdentity
} from '$lib/services/stream-resume.service';

describe('stream-resume.service', () => {
	beforeEach(() => {
		localStorage.clear();
	});
	afterEach(() => {
		localStorage.clear();
	});

	it('returns null when no state exists for the conversation', () => {
		expect(getStreamState('conv-a')).toBeNull();
	});

	it('saves and reads back the byte count', () => {
		saveStreamState('conv-a', 4242);
		const got = getStreamState('conv-a');
		expect(got).not.toBeNull();
		expect(got!.bytesReceived).toBe(4242);
		expect(typeof got!.updatedAt).toBe('number');
	});

	it('overwrites the previous byte count on a new save for the same conversation', () => {
		saveStreamState('conv-a', 100);
		saveStreamState('conv-a', 200);
		const got = getStreamState('conv-a');
		expect(got!.bytesReceived).toBe(200);
	});

	it('keeps states for distinct conversations isolated', () => {
		saveStreamState('conv-a', 10);
		saveStreamState('conv-b', 20);
		expect(getStreamState('conv-a')!.bytesReceived).toBe(10);
		expect(getStreamState('conv-b')!.bytesReceived).toBe(20);
	});

	it('clears the state for a given conversation', () => {
		saveStreamState('conv-a', 10);
		clearStreamState('conv-a');
		expect(getStreamState('conv-a')).toBeNull();
	});

	it('ignores empty conversation id on save', () => {
		saveStreamState('', 1);
		expect(getStreamState('')).toBeNull();
	});

	it('returns null on corrupted storage payload', () => {
		localStorage.setItem('llamacpp.stream.resume.conv-a', '{not-json');
		expect(getStreamState('conv-a')).toBeNull();
	});

	it('persists the model alongside the byte count', () => {
		saveStreamState('conv-a', 10, 'model-x');
		expect(getStreamState('conv-a')!.model).toBe('model-x');
	});

	it('stores a null model when none is provided', () => {
		saveStreamState('conv-a', 10);
		expect(getStreamState('conv-a')!.model).toBeNull();
	});

	it('overwrites the model on a new save for the same conversation', () => {
		saveStreamState('conv-a', 10, 'model-x');
		saveStreamState('conv-a', 20, 'model-y');
		expect(getStreamState('conv-a')!.model).toBe('model-y');
	});

	describe('resumeStreamIdentity', () => {
		it('appends the persisted model so the resume key matches the frozen POST identity', () => {
			saveStreamState('conv-a', 10, 'model-x');
			expect(resumeStreamIdentity('conv-a', getStreamState('conv-a'), 'dropdown')).toBe(
				'conv-a::model-x'
			);
		});

		it('keeps the bare conv id when the persisted model is null', () => {
			saveStreamState('conv-a', 10);
			expect(resumeStreamIdentity('conv-a', getStreamState('conv-a'), 'dropdown')).toBe('conv-a');
		});

		it('falls back to the current model only when no state is persisted', () => {
			expect(resumeStreamIdentity('conv-a', null, 'dropdown')).toBe('conv-a::dropdown');
		});

		it('ignores the fallback when a state exists, the persisted value is authoritative', () => {
			saveStreamState('conv-a', 10, 'model-x');
			expect(resumeStreamIdentity('conv-a', getStreamState('conv-a'), 'dropdown')).toBe(
				'conv-a::model-x'
			);
		});

		it('falls back when a legacy state has no model field', () => {
			localStorage.setItem(
				'llamacpp.stream.resume.conv-a',
				JSON.stringify({ bytesReceived: 10, updatedAt: 1 })
			);
			expect(resumeStreamIdentity('conv-a', getStreamState('conv-a'), 'dropdown')).toBe(
				'conv-a::dropdown'
			);
		});
	});
});
