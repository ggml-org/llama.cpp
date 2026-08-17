import { DISABLED_SKILL_IDS_LOCALSTORAGE_KEY } from '$lib/constants';
import { SkillAvailabilityStore } from '$lib/stores/skill-availability.svelte';
import type { SkillCatalogEntry } from '$lib/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// The node unit project has no DOM or localStorage; install a Map-backed
// Storage and force `browser: true` for this module graph so the constructor
// path (persisted disabled-ID restore) is exercised. A write counter proves
// that repeated no-op setEnabled calls never rewrite localStorage.
const storageFixture = vi.hoisted(() => {
	const state = new Map<string, string>();
	const writes = { count: 0 };

	const polyfill: Storage = {
		clear: () => state.clear(),
		getItem: (key) => state.get(key) ?? null,
		key: (index) => [...state.keys()][index] ?? null,
		get length() {
			return state.size;
		},
		removeItem: (key) => {
			state.delete(key);
		},
		setItem: (key, value) => {
			writes.count += 1;
			state.set(key, String(value));
		}
	};

	return { state, writes, polyfill };
});

vi.mock('$app/environment', () => ({ browser: true }));

// Install the storage polyfill before module scope so the production
// singleton (constructed at import) reads a defined global.
const nodeGlobal = globalThis as unknown as { localStorage: Storage };
nodeGlobal.localStorage = storageFixture.polyfill;

function catalogEntry(id: string, name: string): SkillCatalogEntry {
	return {
		id,
		name,
		description: `description for ${name}`,
		scope: 'global',
		provider: 'test-provider',
		instruction: {
			bytes: 10,
			lines: 1,
			tokens: 4,
			tokens_estimated: false,
			modified_at: null
		},
		resources: { count: 0, truncated: false },
		catalog_xml: `<skills_catalog><skill name="${name}"/></skills_catalog>`
	};
}

beforeEach(() => {
	storageFixture.state.clear();
	storageFixture.writes.count = 0;
});

describe('SkillAvailabilityStore persistence', () => {
	it('loads a valid stored string array of opaque IDs into the disabled set', () => {
		storageFixture.state.set(
			DISABLED_SKILL_IDS_LOCALSTORAGE_KEY,
			JSON.stringify(['entry-a', 'entry-c'])
		);

		const store = new SkillAvailabilityStore();

		expect(store.isDisabled('entry-a')).toBe(true);
		expect(store.isDisabled('entry-c')).toBe(true);
		expect(store.isDisabled('entry-b')).toBe(false);
		expect([...store.disabledIds]).toEqual(['entry-a', 'entry-c']);
	});

	it('falls back to an empty set on malformed JSON, matching persisted() semantics', () => {
		storageFixture.state.set(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY, '{not-json');

		const store = new SkillAvailabilityStore();

		expect(store.disabledIds.size).toBe(0);
		expect(store.isDisabled('entry-a')).toBe(false);
	});

	it('normalizes non-array values and arrays containing non-strings to an empty set', () => {
		const badShapes = ['{"a":1}', '[1, 2]', '[null]', '7', '"entry-a"'];

		for (const serialized of badShapes) {
			storageFixture.state.set(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY, serialized);

			const store = new SkillAvailabilityStore();

			expect(store.disabledIds.size, `shape ${serialized}`).toBe(0);
			expect(
				store.enabledEntries([catalogEntry('entry-a', 'Alpha')]),
				`shape ${serialized}`
			).toEqual([catalogEntry('entry-a', 'Alpha')]);
		}
	});

	it('disabling and re-enabling persist the opaque ID, never the name/scope/provider/path', () => {
		const store = new SkillAvailabilityStore();

		store.setEnabled('opaque-id-1', false);

		expect(store.isDisabled('opaque-id-1')).toBe(true);
		expect(JSON.parse(storageFixture.state.get(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY) ?? '[]')).toEqual([
			'opaque-id-1'
		]);

		store.setEnabled('opaque-id-1', true);

		expect(store.isDisabled('opaque-id-1')).toBe(false);
		expect(JSON.parse(storageFixture.state.get(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY) ?? '[]')).toEqual(
			[]
		);
	});

	it('keeps same-name entries with different IDs independent', () => {
		const store = new SkillAvailabilityStore();

		store.setEnabled('id-shared-a', false);
		store.setEnabled('id-shared-b', false);
		store.setEnabled('id-shared-a', true);

		expect(store.isDisabled('id-shared-a')).toBe(false);
		expect(store.isDisabled('id-shared-b')).toBe(true);
		expect(JSON.parse(storageFixture.state.get(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY) ?? '[]')).toEqual([
			'id-shared-b'
		]);
	});

	it('leaves unknown stored IDs harmlessly persisted', () => {
		storageFixture.state.set(
			DISABLED_SKILL_IDS_LOCALSTORAGE_KEY,
			JSON.stringify(['known-id', 'mystery-id'])
		);

		const store = new SkillAvailabilityStore();

		store.setEnabled('known-id', true);
		expect(JSON.parse(storageFixture.state.get(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY) ?? '[]')).toEqual([
			'mystery-id'
		]);

		store.setEnabled('other-id', false);
		expect(JSON.parse(storageFixture.state.get(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY) ?? '[]')).toEqual([
			'mystery-id',
			'other-id'
		]);
	});
});

describe('SkillAvailabilityStore enabledEntries', () => {
	it('returns the original entries reference when nothing is disabled, preserving server order', () => {
		const entries: readonly SkillCatalogEntry[] = [
			catalogEntry('id-z', 'Zulu'),
			catalogEntry('id-a', 'Alpha'),
			catalogEntry('id-m', 'Mike')
		];

		const store = new SkillAvailabilityStore();

		const result = store.enabledEntries(entries);

		expect(result).toBe(entries);
		expect(result.map((e) => e.id)).toEqual(['id-z', 'id-a', 'id-m']);
	});

	it('filters disabled entries while preserving server order', () => {
		const entries: readonly SkillCatalogEntry[] = [
			catalogEntry('id-z', 'Zulu'),
			catalogEntry('id-a', 'Alpha'),
			catalogEntry('id-m', 'Mike')
		];

		const store = new SkillAvailabilityStore();
		store.setEnabled('id-a', false);

		const result = store.enabledEntries(entries);

		expect(result.map((e) => e.id)).toEqual(['id-z', 'id-m']);
	});
});

describe('SkillAvailabilityStore write discipline', () => {
	it('does not rewrite localStorage on repeated no-op setEnabled calls', () => {
		const store = new SkillAvailabilityStore();

		store.setEnabled('entry-a', false);
		expect(storageFixture.writes.count).toBe(1);
		expect(store.isDisabled('entry-a')).toBe(true);

		store.setEnabled('entry-a', false);
		expect(storageFixture.writes.count).toBe(1);

		store.setEnabled('entry-a', true);
		expect(storageFixture.writes.count).toBe(2);
		expect(store.isDisabled('entry-a')).toBe(false);

		store.setEnabled('entry-a', true);
		expect(storageFixture.writes.count).toBe(2);
	});

	it('does not write when enabling an id that was never disabled', () => {
		const store = new SkillAvailabilityStore();

		store.setEnabled('never-disabled', true);

		expect(storageFixture.writes.count).toBe(0);
		expect(storageFixture.state.has(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY)).toBe(false);
	});
});