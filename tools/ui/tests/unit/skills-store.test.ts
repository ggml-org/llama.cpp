import { skillsStore, type SkillAvailability } from '$lib/stores/skills.svelte';
import type { SkillCatalogEntry, SkillCatalogResponse } from '$lib/types';
import { afterEach, describe, expect, it, vi } from 'vitest';

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'content-type': 'application/json' },
		status
	});
}

function makeEntry(name: string): SkillCatalogEntry {
	return {
		id: `opaque-${name}`,
		name,
		description: `description of ${name}`,
		scope: 'project',
		provider: 'agents',
		instruction: { bytes: 16, lines: 1, tokens: 4, tokens_estimated: true, modified_at: null },
		resources: { count: 0, truncated: false },
		catalog_xml: `<skill><name>${name}</name></skill>`
	};
}

function makeCatalog(...names: string[]): SkillCatalogResponse {
	return {
		skills: names.map(makeEntry),
		catalog_instruction_xml: '<available_skills>Call read_skill(name) when matching.</available_skills>',
		diagnostics: []
	};
}

/** Keys exercised by this suite; every test clears them in afterEach. */
const TEST_CWDS = [undefined, '/a', '/b'] as const;

describe('skillsStore', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
		for (const cwd of TEST_CWDS) {
			skillsStore.invalidate(cwd);
		}

		// Reset the singleton probe gate so the initial-unknown assertion is
		// independent of test order. Test-only: no production reset exists.
		const store = skillsStore as unknown as {
			_availability: SkillAvailability;
			_probeGeneration: number;
		};

		store._availability = 'unknown';
		store._probeGeneration = 0;
	});

	it('keeps the latest catalog per selected CWD and returns each result to its issuer', async () => {
		vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(makeCatalog('alpha'))));

		const result = await skillsStore.refresh('/a');

		expect(result.skills.map((s) => s.name)).toEqual(['alpha']);
		expect(skillsStore.slotFor('/a')?.status).toBe('ready');
		expect(skillsStore.slotFor('/a')?.catalog?.skills.map((s) => s.name)).toEqual(['alpha']);
		expect(skillsStore.slotFor('/b')).toBeUndefined();
	});

	it('keys the no-CWD screen state separately from any selected CWD', async () => {
		vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(makeCatalog('server-cwd'))));

		await skillsStore.refresh(undefined);

		expect(skillsStore.slotFor(undefined)?.status).toBe('ready');
		expect(skillsStore.slotFor('/a')).toBeUndefined();
	});

	it('discards a stale response from the slot but still returns it to its caller', async () => {
		let resolveFirst!: (response: Response) => void;
		let resolveSecond!: (response: Response) => void;
		const fetchMock = vi
			.fn()
			.mockImplementationOnce(
				() =>
					new Promise<Response>((resolve) => {
						resolveFirst = resolve;
					})
			)
			.mockImplementationOnce(
				() =>
					new Promise<Response>((resolve) => {
						resolveSecond = resolve;
					})
			);

		vi.stubGlobal('fetch', fetchMock);

		const first = skillsStore.refresh('/a');
		const second = skillsStore.refresh('/a');

		// The newer request resolves first...
		resolveSecond(jsonResponse(makeCatalog('fresh')));
		const secondResult = await second;
		const freshGeneration = skillsStore.slotFor('/a')?.generation;

		// ...then the older one lands late and must not displace the slot.
		resolveFirst(jsonResponse(makeCatalog('stale')));
		const firstResult = await first;

		expect(secondResult.skills.map((s) => s.name)).toEqual(['fresh']);
		expect(firstResult.skills.map((s) => s.name)).toEqual(['stale']);
		expect(skillsStore.slotFor('/a')?.catalog?.skills.map((s) => s.name)).toEqual(['fresh']);
		expect(skillsStore.slotFor('/a')?.generation).toBe(freshGeneration);
	});

	it('creates a run snapshot from the run own request result, never the mutable slot', async () => {
		const fetchMock = vi
			.fn()
			.mockResolvedValueOnce(jsonResponse(makeCatalog('screen')))
			.mockResolvedValueOnce(jsonResponse(makeCatalog('run')));

		vi.stubGlobal('fetch', fetchMock);

		await skillsStore.refresh('/a');

		const snapshot = await skillsStore.createRunSnapshot('/a');

		expect(snapshot.cwd).toBe('/a');
		expect(snapshot.catalog.skills.map((s) => s.name)).toEqual(['run']);
		expect(snapshot.entries.map((e) => e.name)).toEqual(['run']);
		expect(snapshot.envelope).toContain('<skill><name>run</name></skill>');
		expect(snapshot.envelope).not.toContain('<skill><name>screen</name></skill>');
		// The screen slot is untouched by the run's own request.
		expect(skillsStore.slotFor('/a')?.catalog?.skills.map((s) => s.name)).toEqual(['screen']);
	});

	it('keeps a frozen snapshot stable across later store refreshes', async () => {
		const fetchMock = vi
			.fn()
			.mockResolvedValueOnce(jsonResponse(makeCatalog('first')))
			.mockResolvedValueOnce(jsonResponse(makeCatalog('second')));

		vi.stubGlobal('fetch', fetchMock);

		const snapshot = await skillsStore.createRunSnapshot('/a');

		await skillsStore.refresh('/a');

		expect(snapshot.entries.map((e) => e.name)).toEqual(['first']);
		expect(snapshot.envelope).toContain('first');
		expect(snapshot.envelope).not.toContain('second');
	});

	it('invalidates the screen slot and stales any in-flight response for that CWD', async () => {
		let resolvePending!: (response: Response) => void;
		const fetchMock = vi.fn(
			() =>
				new Promise<Response>((resolve) => {
					resolvePending = resolve;
				})
		);

		vi.stubGlobal('fetch', fetchMock);

		const pending = skillsStore.refresh('/a');

		skillsStore.invalidate('/a');

		expect(skillsStore.slotFor('/a')).toBeUndefined();

		resolvePending(jsonResponse(makeCatalog('late')));
		const result = await pending;

		expect(result.skills.map((s) => s.name)).toEqual(['late']);
		expect(skillsStore.slotFor('/a')).toBeUndefined();
	});

	it('records request failures in the slot and rethrows them to the issuer', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue(jsonResponse({ error: { code: 503, message: 'unavailable' } }, 503))
		);

		await expect(skillsStore.refresh('/a')).rejects.toMatchObject({
			name: 'ApiError',
			status: 503
		});
		expect(skillsStore.slotFor('/a')?.status).toBe('error');
	});

	it('starts unknown and hidden, then one shared probe makes the catalog available', async () => {
		expect(skillsStore.availability).toBe('unknown');
		expect(skillsStore.showInNavigation).toBe(false);

		const fetchMock = vi.fn().mockResolvedValue(jsonResponse(makeCatalog('alpha')));

		vi.stubGlobal('fetch', fetchMock);

		const first = skillsStore.probeAvailability(undefined);
		const second = skillsStore.probeAvailability(undefined);

		expect(fetchMock).toHaveBeenCalledTimes(1);

		await expect(Promise.all([first, second])).resolves.toHaveLength(2);

		expect(skillsStore.availability).toBe('available');
		expect(skillsStore.showInNavigation).toBe(true);
		// A successful probe leaves a ready slot the route can consume.
		expect(skillsStore.slotFor(undefined)?.status).toBe('ready');
		expect(skillsStore.slotFor(undefined)?.catalog?.skills.map((s) => s.name)).toEqual(['alpha']);
	});

	it('maps a 404 probe to disabled and hides the navigation entry', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue(jsonResponse({ error: { code: 404, message: 'no skills route' } }, 404))
		);

		await skillsStore.probeAvailability(undefined);

		expect(skillsStore.availability).toBe('disabled');
		expect(skillsStore.showInNavigation).toBe(false);
		expect(skillsStore.slotFor(undefined)?.status).toBe('error');
	});

	it('maps a 503 probe to error and keeps the navigation entry visible', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue(jsonResponse({ error: { code: 503, message: 'unavailable' } }, 503))
		);

		await skillsStore.probeAvailability(undefined);

		expect(skillsStore.availability).toBe('error');
		expect(skillsStore.showInNavigation).toBe(true);
		expect(skillsStore.slotFor(undefined)?.status).toBe('error');
	});

	it('coalesces concurrent ensureCatalog callers only for the same CWD', async () => {
		const fetchMock = vi.fn(async () => jsonResponse(makeCatalog('alpha')));

		vi.stubGlobal('fetch', fetchMock);

		const [a1, a2] = await Promise.all([
			skillsStore.ensureCatalog('/a'),
			skillsStore.ensureCatalog('/a')
		]);

		expect(a1.skills.map((s) => s.name)).toEqual(['alpha']);
		expect(a2.skills.map((s) => s.name)).toEqual(['alpha']);
		expect(fetchMock).toHaveBeenCalledTimes(1);

		const [b1] = await Promise.all([skillsStore.ensureCatalog('/b')]);

		expect(b1.skills.map((s) => s.name)).toEqual(['alpha']);
		expect(fetchMock).toHaveBeenCalledTimes(2);
	});

	it('gives each ensureCatalog caller independent abort behavior and aborts the store request after the last subscriber leaves', async () => {
		const pending: Array<{ resolve: (response: Response) => void; signal: AbortSignal }> = [];
		const fetchMock = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
			const { promise, resolve } = Promise.withResolvers<Response>();

			pending.push({ resolve, signal: init?.signal ?? new AbortController().signal });

			return promise;
		});

		vi.stubGlobal('fetch', fetchMock);

		const controllerA = new AbortController();
		const controllerB = new AbortController();

		const a = skillsStore.ensureCatalog('/a', controllerA.signal);
		const b = skillsStore.ensureCatalog('/a', controllerB.signal);

		expect(pending).toHaveLength(1);
		expect(pending[0].signal.aborted).toBe(false);

		controllerA.abort();
		await expect(a).rejects.toThrow();
		// B is still attached, so the shared request stays alive.
		expect(pending[0].signal.aborted).toBe(false);
		await expect(Promise.race([b, Promise.resolve('still-pending')])).resolves.toBe('still-pending');

		controllerB.abort();
		await expect(b).rejects.toThrow();
		// The final subscriber left, so the store-owned request is aborted.
		expect(pending[0].signal.aborted).toBe(true);
	});

	it('does not change availability when the probing caller aborts', async () => {
		vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));

		const controller = new AbortController();
		const probe = skillsStore.probeAvailability(undefined, controller.signal);

		expect(skillsStore.availability).toBe('loading');

		controller.abort();
		await expect(probe).rejects.toThrow();

		expect(skillsStore.availability).toBe('loading');
		expect(skillsStore.showInNavigation).toBe(false);
	});
});
