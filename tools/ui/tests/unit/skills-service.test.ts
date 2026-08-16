import { SkillsService } from '$lib/services/skills.service';
import type { SkillCatalogResponse, SkillReadResult } from '$lib/types';
import { afterEach, describe, expect, it, vi } from 'vitest';

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'content-type': 'application/json' },
		status
	});
}

function makeCatalog(): SkillCatalogResponse {
	return {
		catalog_instruction_xml:
			'<available_skills>Call read_skill(name) when a task matches a skill description.</available_skills>',
		diagnostics: [],
		skills: [
			{
				catalog_xml:
					'<skill><name>example-skill</name><description>Use when processing example inputs.</description></skill>',
				description: 'Use when processing example inputs.',
				id: 'opaque-resolved-skill-identity',
				instruction: {
					bytes: 4096,
					lines: 128,
					modified_at: '2026-08-11T12:34:56Z',
					tokens: 1024,
					tokens_estimated: false
				},
				name: 'example-skill',
				provider: 'agents',
				resources: { count: 2, truncated: false },
				scope: 'project'
			}
		]
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
			const fetchMock = vi.fn().mockResolvedValue(jsonResponse(makeBaseReadResult()));

			vi.stubGlobal('fetch', fetchMock);

			await SkillsService.read({ name: 'example-skill', path: 'references/DETAILS.md' }, '/w');

			const [, init] = fetchMock.mock.calls[0];
			const body = JSON.parse(init.body as string) as Record<string, unknown>;

			expect(body).toEqual({ name: 'example-skill', path: 'references/DETAILS.md' });
			expect(Object.keys(body).sort()).toEqual(['name', 'path']);
			expect((init.headers as Record<string, string>)['x-skill-cwd']).toBe('/w');
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

		it('returns the typed base read result', async () => {
			vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(makeBaseReadResult())));

			const result = await SkillsService.read({ name: 'example-skill' });

			expect(result.kind).toBe('skill');

			if (result.kind === 'skill') {
				expect(result.source).toBe(BASE_SKILL_SOURCE);
				expect(result.body_markdown).toBe(BASE_SKILL_BODY_MARKDOWN);
				expect(result.content_xml).toBe(makeBaseReadResult().content_xml);
				expect(result.skill.id).toBe('opaque-resolved-skill-identity');
			}
		});
	});
});
