// Guards the route-owned catalog refresh lifecycle: initial CWD fetch,
// immediate slot invalidation + abortable refresh on CWD change, stale
// response suppression across the CWD switch, retry, and unmount dispose.
// Frozen agent run snapshots are never touched (store-level guarantee).

import { useSkillCatalogRefresh } from '$lib/hooks/use-skill-catalog-refresh.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import type { SkillCatalogEntry, SkillCatalogResponse } from '$lib/types';
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest';

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

/** One captured fetch call; the test resolves it later to control ordering. */
interface PendingRequest {
	url: string;
	headers: Record<string, string>;
	signal: AbortSignal;
	resolve: (response: Response) => void;
}

/**
 * Requests a test captured but never resolved (so the shared store entry stays
 * live). afterEach releases them so a settled entry cannot leak into the next
 * test's deduplication decision.
 */
const liveRequests: Array<() => void> = [];

/** Stub fetch to capture requests without resolving them until asked. */
function captureFetch(pending: PendingRequest[]): Mock {
	const fetchMock = vi.fn(
		(input: RequestInfo | URL, init?: RequestInit) =>
			new Promise<Response>((resolve) => {
				pending.push({
					headers: (init?.headers ?? {}) as Record<string, string>,
					resolve,
					signal: init?.signal ?? new AbortController().signal,
					url: String(input)
				});
				liveRequests.push(() => resolve(jsonResponse(makeCatalog('released'))));
			})
	);

	vi.stubGlobal('fetch', fetchMock);

	return fetchMock;
}

/** Keys exercised by this suite; every test clears them in afterEach. */
const TEST_CWDS = [undefined, '/a', '/b'] as const;

describe('useSkillCatalogRefresh', () => {
	afterEach(async () => {
		// Settle any still-pending captured request so its shared store entry
		// is removed before the next test probes the same CWD.
		for (const release of liveRequests) release();
		liveRequests.length = 0;
		const { promise, resolve } = Promise.withResolvers<void>();

		setTimeout(resolve, 0);
		await promise;

		vi.restoreAllMocks();
		vi.unstubAllGlobals();
		for (const cwd of TEST_CWDS) {
			skillsStore.invalidate(cwd);
		}
	});

	it('fetches the catalog for the selected CWD on the first change', async () => {
		const pending: PendingRequest[] = [];

		captureFetch(pending);
		const refresh = useSkillCatalogRefresh();

		refresh.onCwdChange('/a');

		expect(pending).toHaveLength(1);
		expect(pending[0].url).toBe('/skills');
		expect(pending[0].headers).toMatchObject({ 'x-skill-cwd': '/a' });
		expect(skillsStore.slotFor('/a')?.status).toBe('loading');

		pending[0].resolve(jsonResponse(makeCatalog('alpha')));

		await vi.waitFor(() => expect(skillsStore.slotFor('/a')?.status).toBe('ready'));
		expect(skillsStore.slotFor('/a')?.catalog?.skills.map((s) => s.name)).toEqual(['alpha']);
	});

	it('omits the CWD header when no working directory is selected', () => {
		const pending: PendingRequest[] = [];

		captureFetch(pending);
		const refresh = useSkillCatalogRefresh();

		refresh.onCwdChange(undefined);

		expect(pending).toHaveLength(1);
		expect('x-skill-cwd' in pending[0].headers).toBe(false);
	});

	it('invalidates the previous slot and aborts its request on CWD change', async () => {
		const pending: PendingRequest[] = [];

		captureFetch(pending);
		const refresh = useSkillCatalogRefresh();

		refresh.onCwdChange('/a');
		expect(skillsStore.slotFor('/a')?.status).toBe('loading');

		refresh.onCwdChange('/b');

		expect(pending).toHaveLength(2);
		expect(skillsStore.slotFor('/a')).toBeUndefined();
		expect(pending[0].signal.aborted).toBe(true);
		expect(pending[1].headers).toMatchObject({ 'x-skill-cwd': '/b' });
		expect(skillsStore.slotFor('/b')?.status).toBe('loading');

		pending[1].resolve(jsonResponse(makeCatalog('beta')));

		await vi.waitFor(() => expect(skillsStore.slotFor('/b')?.status).toBe('ready'));
		expect(skillsStore.slotFor('/b')?.catalog?.skills.map((s) => s.name)).toEqual(['beta']);
	});

	it('never lets a late response for the previous CWD replace the new slot', async () => {
		const pending: PendingRequest[] = [];

		captureFetch(pending);
		const refresh = useSkillCatalogRefresh();

		refresh.onCwdChange('/a');
		refresh.onCwdChange('/b');

		// The newer CWD resolves first...
		pending[1].resolve(jsonResponse(makeCatalog('fresh')));

		await vi.waitFor(() => expect(skillsStore.slotFor('/b')?.status).toBe('ready'));

		// ...then the previous CWD's response lands late and must not displace it.
		pending[0].resolve(jsonResponse(makeCatalog('stale')));
		await vi.waitFor(() => expect(skillsStore.slotFor('/a')).toBeUndefined());

		expect(skillsStore.slotFor('/b')?.catalog?.skills.map((s) => s.name)).toEqual(['fresh']);
	});

	it('dispose aborts the in-flight request and makes the controller inert', () => {
		const pending: PendingRequest[] = [];

		captureFetch(pending);
		const refresh = useSkillCatalogRefresh();

		refresh.onCwdChange('/a');
		expect(pending[0].signal.aborted).toBe(false);

		refresh.dispose();
		expect(pending[0].signal.aborted).toBe(true);

		refresh.onCwdChange('/b');
		refresh.retry();
		expect(pending).toHaveLength(1);
	});

	it('retry re-requests the current CWD (error-state refresh)', async () => {
		const pending: PendingRequest[] = [];

		captureFetch(pending);
		const refresh = useSkillCatalogRefresh();

		refresh.onCwdChange('/a');
		pending[0].resolve(jsonResponse(makeCatalog('alpha')));

		await vi.waitFor(() => expect(skillsStore.slotFor('/a')?.status).toBe('ready'));

		refresh.retry();

		expect(pending).toHaveLength(2);
		expect(pending[1].headers).toMatchObject({ 'x-skill-cwd': '/a' });
		expect(skillsStore.slotFor('/a')?.status).toBe('loading');

		pending[1].resolve(jsonResponse(makeCatalog('alpha-2')));

		await vi.waitFor(() => expect(skillsStore.slotFor('/a')?.status).toBe('ready'));
		expect(skillsStore.slotFor('/a')?.catalog?.skills.map((s) => s.name)).toEqual(['alpha-2']);
	});

	it('treats repeated identical CWD changes as no-ops', () => {
		const pending: PendingRequest[] = [];

		captureFetch(pending);
		const refresh = useSkillCatalogRefresh();

		refresh.onCwdChange('/a');
		refresh.onCwdChange('/a');
		refresh.onCwdChange('/a');

		expect(pending).toHaveLength(1);
	});

	it('shares one request between sidebar probing and route loading, while retry forces a new one', async () => {
		const pending: PendingRequest[] = [];

		captureFetch(pending);
		const refresh = useSkillCatalogRefresh();
		const probeController = new AbortController();

		const probe = skillsStore.probeAvailability('/a', probeController.signal);

		refresh.onCwdChange('/a');

		// Sidebar probing and initial route loading for one CWD share one request.
		expect(pending).toHaveLength(1);
		expect(pending[0].headers).toMatchObject({ 'x-skill-cwd': '/a' });

		pending[0].resolve(jsonResponse(makeCatalog('alpha')));

		await Promise.all([
			probe,
			vi.waitFor(() => expect(skillsStore.slotFor('/a')?.status).toBe('ready'))
		]);

		expect(skillsStore.availability).toBe('available');

		// Retry is a forced refresh and always issues a fresh request.
		refresh.retry();
		expect(pending).toHaveLength(2);

		pending[1].resolve(jsonResponse(makeCatalog('alpha-2')));

		await vi.waitFor(() => expect(skillsStore.slotFor('/a')?.catalog?.skills.map((s) => s.name)).toEqual(['alpha-2']));

		probeController.abort();
	});
});
