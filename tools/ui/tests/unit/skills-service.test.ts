// Guards SkillsService transport, the catalog slot keepers, and availability probing.

import { SkillsService } from '$lib/services/skills.service';
import { type SkillAvailability, skillsStore } from '$lib/stores/skills.svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { catalogOf, jsonResponse, resourceResult } from '../fixtures/skills';

const TEST_CWDS = [undefined, '/a'] as const;

describe('SkillsService', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it('GETs /skills with the selected CWD header, or none for a missing or blank CWD', async () => {
		const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(jsonResponse(catalogOf())));

		vi.stubGlobal('fetch', fetchMock);

		await SkillsService.list('/workspace/project');
		await SkillsService.list(undefined);
		await SkillsService.list('   ');

		const headers = fetchMock.mock.calls.map(([, init]) => init.headers as Record<string, string>);

		expect(headers).toEqual([
			{ 'Content-Type': 'application/json', 'x-skill-cwd': '/workspace/project' },
			{ 'Content-Type': 'application/json' },
			{ 'Content-Type': 'application/json' }
		]);
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

			return Promise.resolve(jsonResponse(catalogOf()));
		});

		vi.stubGlobal('fetch', fetchMock);

		await expect(SkillsService.list(undefined, controller.signal)).rejects.toThrow();
		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(fetchMock.mock.calls[0][1]?.signal).toBe(controller.signal);
	});

	it('POSTs /skills/read with only name and the optional path, dropping smuggled fields', async () => {
		const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(jsonResponse(resourceResult())));

		vi.stubGlobal('fetch', fetchMock);

		await SkillsService.read({ name: 'example-skill' });
		await SkillsService.read(
			{
				id: 'client-forged-id',
				name: 'example-skill',
				path: 'references/DETAILS.md',
				provider: 'agents',
				scope: 'project'
			} as unknown as Parameters<typeof SkillsService.read>[0],
			'/w'
		);

		const [first, second] = fetchMock.mock.calls;

		expect(first[1].method).toBe('POST');
		expect(String(first[0])).toContain('/skills/read');
		expect(JSON.parse(first[1].body as string)).toEqual({ name: 'example-skill' });
		expect(JSON.parse(second[1].body as string)).toEqual({
			name: 'example-skill',
			path: 'references/DETAILS.md'
		});
		expect((second[1].headers as Record<string, string>)['x-skill-cwd']).toBe('/w');
	});
});

describe('skillsStore', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
		for (const cwd of TEST_CWDS) {
			skillsStore.invalidate(cwd);
		}

		// Reset the probe gate so tests are order-independent.
		const store = skillsStore as unknown as {
			_availability: SkillAvailability;
			_probeGeneration: number;
		};

		store._availability = 'unknown';
		store._probeGeneration = 0;
	});

	it('discards a stale response from the slot but still returns it to its caller', async () => {
		let resolveFirst!: (response: Response) => void;
		let resolveSecond!: (response: Response) => void;

		vi.stubGlobal(
			'fetch',
			vi
				.fn()
				.mockImplementationOnce(
					() => new Promise<Response>((resolve) => (resolveFirst = resolve))
				)
				.mockImplementationOnce(
					() => new Promise<Response>((resolve) => (resolveSecond = resolve))
				)
		);

		const first = skillsStore.refresh('/a');
		const second = skillsStore.refresh('/a');

		resolveSecond(jsonResponse(catalogOf('fresh')));
		const secondResult = await second;

		resolveFirst(jsonResponse(catalogOf('stale')));
		const firstResult = await first;

		expect(secondResult.skills.map((s) => s.name)).toEqual(['fresh']);
		expect(firstResult.skills.map((s) => s.name)).toEqual(['stale']);
		expect(skillsStore.slotFor('/a')?.catalog?.skills.map((s) => s.name)).toEqual(['fresh']);
	});

	it('creates a run snapshot from the run own request result, never the mutable slot', async () => {
		vi.stubGlobal(
			'fetch',
			vi
				.fn()
				.mockResolvedValueOnce(jsonResponse(catalogOf('screen')))
				.mockResolvedValueOnce(jsonResponse(catalogOf('run')))
		);

		await skillsStore.refresh('/a');

		const snapshot = await skillsStore.createRunSnapshot('/a');

		expect(snapshot.cwd).toBe('/a');
		expect(snapshot.entries.map((e) => e.name)).toEqual(['run']);
		expect(snapshot.envelope).toContain('<skill><name>run</name></skill>');
		expect(snapshot.envelope).not.toContain('<skill><name>screen</name></skill>');
		// The screen slot is untouched by the run's own request.
		expect(skillsStore.slotFor('/a')?.catalog?.skills.map((s) => s.name)).toEqual(['screen']);
	});

	it('keeps a frozen snapshot stable across later store refreshes', async () => {
		vi.stubGlobal(
			'fetch',
			vi
				.fn()
				.mockResolvedValueOnce(jsonResponse(catalogOf('first')))
				.mockResolvedValueOnce(jsonResponse(catalogOf('second')))
		);

		const snapshot = await skillsStore.createRunSnapshot('/a');

		await skillsStore.refresh('/a');

		expect(snapshot.entries.map((e) => e.name)).toEqual(['first']);
		expect(snapshot.envelope).toContain('first');
		expect(snapshot.envelope).not.toContain('second');
	});

	it.each([
		['available', 200, catalogOf('alpha'), true],
		['disabled', 404, { error: { code: 404, message: 'no skills route' } }, false],
		['error', 503, { error: { code: 503, message: 'unavailable' } }, true]
	] as const)(
		'maps a probe response to %s availability',
		async (expected, status, body, navigationVisible) => {
			vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(body, status)));

			await skillsStore.probeAvailability(undefined);

			expect(skillsStore.availability).toBe(expected);
			expect(skillsStore.showInNavigation).toBe(navigationVisible);
		}
	);

	it('shares one probe request across concurrent callers and leaves a ready slot', async () => {
		const fetchMock = vi.fn().mockResolvedValue(jsonResponse(catalogOf('alpha')));

		vi.stubGlobal('fetch', fetchMock);

		await expect(
			Promise.all([skillsStore.probeAvailability(undefined), skillsStore.probeAvailability(undefined)])
		).resolves.toHaveLength(2);

		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(skillsStore.availability).toBe('available');
		expect(skillsStore.slotFor(undefined)?.status).toBe('ready');
		expect(skillsStore.slotFor(undefined)?.catalog?.skills.map((s) => s.name)).toEqual(['alpha']);
	});
});
