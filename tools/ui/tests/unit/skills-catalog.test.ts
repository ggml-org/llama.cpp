import { useSkillCatalogRefresh } from '$lib/hooks/use-skill-catalog-refresh.svelte';
import { SkillsService } from '$lib/services/skills.service';
import { skillAvailabilityStore } from '$lib/stores/skill-availability.svelte';
import { type SkillAvailability, skillsStore } from '$lib/stores/skills.svelte';
import type { SkillCatalogEntry, SkillCatalogResponse, SkillReadResult } from '$lib/types';
import { afterEach, describe, expect, it, type Mock, vi } from 'vitest';

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'content-type': 'application/json' },
		status
	});
}

function makeEntry(name: string): SkillCatalogEntry {
	return {
		catalog_xml: `<skill><name>${name}</name></skill>`,
		description: `description of ${name}`,
		id: `opaque-${name}`,
		instruction: { bytes: 16, lines: 1, modified_at: null, tokens: 4, tokens_estimated: true },
		name,
		provider: 'agents',
		resources: { count: 0, truncated: false },
		scope: 'project'
	};
}

function makeCatalog(...names: string[]): SkillCatalogResponse {
	return {
		catalog_instruction_xml:
			'<available_skills>Call read_skill(name) when matching.</available_skills>',
		diagnostics: [],
		skills: names.map(makeEntry)
	};
}

const BASE_SKILL_SOURCE =
	'---\nname: example-skill\ndescription: Use when processing example inputs.\n---\n# Example\n\nUse **carefully**.\n';
const BASE_SKILL_BODY_MARKDOWN = '# Example\n\nUse **carefully**.\n';

function makeBaseReadResult(): SkillReadResult {
	return {
		body_markdown: BASE_SKILL_BODY_MARKDOWN,
		content_xml:
			'<skill_content name="example-skill"><skill_resources><file>references/DETAILS.md</file></skill_resources></skill_content>',
		diagnostics: [],
		kind: 'skill',
		resources: { paths: ['references/DETAILS.md'], truncated: false },
		skill: {
			id: 'opaque-resolved-skill-identity',
			metadata: { description: 'Use when processing example inputs.' },
			name: 'example-skill',
			provider: 'agents',
			scope: 'project'
		},
		source: BASE_SKILL_SOURCE
	};
}

const RESOURCE_SOURCE = '# Resource\n\nExact bytes as stored.\n';

function makeResourceReadResult(): SkillReadResult {
	return {
		content_xml:
			'<skill_resource name="example-skill" path="references/DETAILS.md"># Resource\n\nExact bytes as stored.\n</skill_resource>',
		diagnostics: [],
		kind: 'resource',
		resource: { path: 'references/DETAILS.md' },
		skill: {
			id: 'opaque-resolved-skill-identity',
			name: 'example-skill',
			provider: 'agents',
			scope: 'project'
		},
		source: RESOURCE_SOURCE
	};
}

/** Keys exercised by the store and hook suites; every test clears them in afterEach. */
const TEST_CWDS = [undefined, '/a', '/b'] as const;

describe('SkillsService', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	describe('list', () => {
		it('GETs /skills without an X-Skill-Cwd header when no CWD is selected', async () => {
			const fetchMock = vi.fn().mockResolvedValue(jsonResponse(makeCatalog()));

			vi.stubGlobal('fetch', fetchMock);

			await SkillsService.list();

			expect(fetchMock).toHaveBeenCalledTimes(1);

			const [url, init] = fetchMock.mock.calls[0];

			expect(String(url)).toContain('/skills');
			expect((init.method ?? 'GET') as string).toBe('GET');
			expect((init.headers as Record<string, string>)['x-skill-cwd']).toBeUndefined();
		});

		it('sends the selected CWD as the X-Skill-Cwd header', async () => {
			const fetchMock = vi.fn().mockResolvedValue(jsonResponse(makeCatalog()));

			vi.stubGlobal('fetch', fetchMock);

			await SkillsService.list('/workspace/project');

			const [, init] = fetchMock.mock.calls[0];

			expect((init.headers as Record<string, string>)['x-skill-cwd']).toBe('/workspace/project');
		});

		it('omits the header for a whitespace-only selected CWD', async () => {
			const fetchMock = vi.fn().mockResolvedValue(jsonResponse(makeCatalog()));

			vi.stubGlobal('fetch', fetchMock);

			await SkillsService.list('   ');

			const [, init] = fetchMock.mock.calls[0];

			expect((init.headers as Record<string, string>)['x-skill-cwd']).toBeUndefined();
		});

		it('propagates handler errors as ApiError with the status code', async () => {
			vi.stubGlobal(
				'fetch',
				vi
					.fn()
					.mockResolvedValue(
						jsonResponse(
							{ error: { code: 400, message: 'Invalid CWD', type: 'invalid_request_error' } },
							400
						)
					)
			);

			await expect(SkillsService.list('/bad')).rejects.toMatchObject({
				name: 'ApiError',
				status: 400
			});
		});

		it('propagates an aborted request as a rejection and passes the signal to fetch', async () => {
			const controller = new AbortController();

			controller.abort();

			const fetchMock = vi.fn((_url: RequestInfo | URL, init?: RequestInit) => {
				if (init?.signal?.aborted) {
					return Promise.reject(new DOMException('This operation was aborted', 'AbortError'));
				}

				return Promise.resolve(jsonResponse(makeCatalog()));
			});

			vi.stubGlobal('fetch', fetchMock);

			await expect(SkillsService.list(undefined, controller.signal)).rejects.toThrow();
			expect(fetchMock).toHaveBeenCalledTimes(1);
			expect(fetchMock.mock.calls[0][1]?.signal).toBe(controller.signal);
		});
	});

	describe('read', () => {
		it('POSTs /skills/read with only the name when no path is given', async () => {
			const fetchMock = vi.fn().mockResolvedValue(jsonResponse(makeBaseReadResult()));

			vi.stubGlobal('fetch', fetchMock);

			await SkillsService.read({ name: 'example-skill' });

			const [url, init] = fetchMock.mock.calls[0];

			expect(String(url)).toContain('/skills/read');
			expect(init.method).toBe('POST');
			expect(JSON.parse(init.body as string)).toEqual({ name: 'example-skill' });
			expect((init.headers as Record<string, string>)['x-skill-cwd']).toBeUndefined();
		});

		it('POSTs the optional path and nothing else', async () => {
			const fetchMock = vi.fn().mockResolvedValue(jsonResponse(makeResourceReadResult()));

			vi.stubGlobal('fetch', fetchMock);

			const result = await SkillsService.read(
				{ name: 'example-skill', path: 'references/DETAILS.md' },
				'/w'
			);

			const [, init] = fetchMock.mock.calls[0];
			const body = JSON.parse(init.body as string) as Record<string, unknown>;

			expect(body).toEqual({ name: 'example-skill', path: 'references/DETAILS.md' });
			expect(Object.keys(body).sort()).toEqual(['name', 'path']);
			expect((init.headers as Record<string, string>)['x-skill-cwd']).toBe('/w');
			expect(result.source).toBe(RESOURCE_SOURCE);
		});

		it('never forwards client identity or origin fields', async () => {
			const fetchMock = vi.fn().mockResolvedValue(jsonResponse(makeBaseReadResult()));

			vi.stubGlobal('fetch', fetchMock);

			// Runtime attackers may stuff extra fields; the transport must drop them.
			const smuggled = {
				id: 'client-forged-id',
				name: 'example-skill',
				path: 'references/DETAILS.md',
				provider: 'agents',
				scope: 'project'
			} as unknown as Parameters<typeof SkillsService.read>[0];

			await SkillsService.read(smuggled);

			const [, init] = fetchMock.mock.calls[0];

			expect(JSON.parse(init.body as string)).toEqual({
				name: 'example-skill',
				path: 'references/DETAILS.md'
			});
		});
	});
});

describe('skillsStore', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
		for (const cwd of TEST_CWDS) {
			skillsStore.invalidate(cwd);
		}

		// Reset the probe gate so this test is order-independent.
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

		// ...then the older one lands late and must not displace the slot.
		resolveFirst(jsonResponse(makeCatalog('stale')));
		const firstResult = await first;

		expect(secondResult.skills.map((s) => s.name)).toEqual(['fresh']);
		expect(firstResult.skills.map((s) => s.name)).toEqual(['stale']);
		expect(skillsStore.slotFor('/a')?.catalog?.skills.map((s) => s.name)).toEqual(['fresh']);
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

	it('applies the current disabled set to the freshly fetched catalog snapshot', async () => {
		vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(makeCatalog('run', 'manual'))));

		// Snapshot filtering uses the freshly fetched catalog IDs.
		skillAvailabilityStore.setEnabled('opaque-manual', false);

		try {
			const snapshot = await skillsStore.createRunSnapshot('/a');

			expect(snapshot.entries.map((e) => e.name)).toEqual(['run']);
			expect(snapshot.envelope).toContain('<skill><name>run</name></skill>');
			expect(snapshot.envelope).not.toContain('<skill><name>manual</name></skill>');
			// The raw browsing catalog still carries the disabled entry.
			expect(snapshot.catalog.skills.map((s) => s.name)).toEqual(['run', 'manual']);
		} finally {
			skillAvailabilityStore.setEnabled('opaque-manual', true);
		}
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
			vi
				.fn()
				.mockResolvedValue(jsonResponse({ error: { code: 404, message: 'no skills route' } }, 404))
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
		await expect(Promise.race([b, Promise.resolve('still-pending')])).resolves.toBe(
			'still-pending'
		);

		controllerB.abort();
		await expect(b).rejects.toThrow();
		// The final subscriber left, so the store-owned request is aborted.
		expect(pending[0].signal.aborted).toBe(true);
	});
});

/**
 * Requests a test captured but never resolved (so the shared store entry stays
 * live). afterEach releases them so a settled entry cannot leak into the next
 * test's deduplication decision.
 */
const liveRequests: Array<() => void> = [];

/** One captured fetch call; the test resolves it later to control ordering. */
interface PendingRequest {
	url: string;
	headers: Record<string, string>;
	signal: AbortSignal;
	resolve: (response: Response) => void;
}

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

		await vi.waitFor(() =>
			expect(skillsStore.slotFor('/a')?.catalog?.skills.map((s) => s.name)).toEqual(['alpha-2'])
		);

		probeController.abort();
	});
});
