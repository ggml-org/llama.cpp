import SkillDetail from '$lib/components/app/skills/SkillDetail.svelte';
import { DatabaseService } from '$lib/services/database.service';
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import type { SkillBaseReadResult, SkillCatalogEntry, SkillResourceReadResult } from '$lib/types';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

const READ_URL = '/skills/read';
const CWD = '/srv/project-a';
const RAW_SOURCE =
	'---\nname: demo-skill\ndescription: raw frontmatter\n---\n# Rendered heading\n\nBody text.\n';
const BODY_MARKDOWN = '# Rendered heading\n\nBody text.\n';

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'content-type': 'application/json' },
		status
	});
}

function makeEntry(name = 'demo-skill', overrides: Partial<SkillCatalogEntry> = {}): SkillCatalogEntry {
	return {
		catalog_xml: `<skill><name>${name}</name></skill>`,
		description: `description of ${name}`,
		id: `opaque-${name}`,
		instruction: { bytes: 16, lines: 1, modified_at: null, tokens: 4, tokens_estimated: true },
		name,
		provider: 'agents',
		resources: { count: 0, truncated: false },
		scope: 'project',
		...overrides
	};
}

function baseResult(overrides: Partial<SkillBaseReadResult> = {}): SkillBaseReadResult {
	return {
		body_markdown: BODY_MARKDOWN,
		content_xml: '<skill_content name="demo-skill">body</skill_content>',
		diagnostics: [],
		kind: 'skill',
		resources: { paths: [], truncated: false },
		skill: {
			id: 'opaque-demo-skill',
			metadata: {
				description: 'Structured metadata description',
				license: 'MIT',
				name: 'demo-skill'
			},
			name: 'demo-skill',
			provider: 'agents',
			scope: 'project'
		},
		source: RAW_SOURCE,
		...overrides
	};
}

function resourceResult(name = 'demo-skill'): SkillResourceReadResult {
	return {
		content_xml: '<skill_resource>data</skill_resource>',
		diagnostics: [],
		kind: 'resource',
		resource: { path: 'refs/DETAILS.md' },
		source: 'data',
		skill: { id: `opaque-${name}`, name, provider: 'agents', scope: 'project' }
	};
}

/** Controllable /skills/read fetch that rejects the in-flight request on abort. */
function deferredRead() {
	const state: { signal?: AbortSignal } = {};
	const { promise, reject, resolve } = Promise.withResolvers<Response>();

	return {
		attach(init?: RequestInit) {
			state.signal = init?.signal ?? undefined;
			init?.signal?.addEventListener(
				'abort',
				() => reject(new DOMException('The operation was aborted.', 'AbortError')),
				{ once: true }
			);
		},
		promise,
		reject,
		resolve,
		get signal() {
			return state.signal;
		}
	};
}

function mockRead(read: (init: RequestInit) => Response | Promise<Response>) {
	vi.mocked(fetch).mockImplementation(async (url, init) => {
		if (String(url).includes(READ_URL)) return read(init as RequestInit);

		return jsonResponse({ catalog_instruction_xml: '', diagnostics: [], skills: [] });
	});
}

function bodyText(): string {
	return document.body.textContent ?? '';
}

afterEach(() => {
	vi.restoreAllMocks();
});

describe('SkillDetail preview', () => {
	it('renders the markdown body by default and keeps the raw frontmatter out of it', async () => {
		mockRead(() => jsonResponse(baseResult()));

		const screen = await render(SkillDetail, {
			props: { cwd: undefined, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		// The detail header maps the `agents` provider value to `generic`.
		expect(bodyText()).toContain('project / generic');

		expect(screen.getByRole('button', { name: 'Markdown' })).toBeTruthy();
		expect(screen.getByRole('button', { name: 'Raw' })).toBeTruthy();

		const markdownPane = screen.getByTestId('skill-detail-markdown').element();

		expect(markdownPane.textContent).toContain('Rendered heading');
		expect(markdownPane.textContent).not.toContain('description: raw frontmatter');

		await screen.getByRole('button', { name: 'Raw' }).click();

		const rawPane = screen.getByTestId('skill-detail-raw').element();

		expect(rawPane.textContent).toContain('---');
		expect(rawPane.textContent).toContain('description: raw frontmatter');
	});

	it('shows a Manual only badge for a flagged entry', async () => {
		mockRead(() => jsonResponse(baseResult()));

		const screen = await render(SkillDetail, {
			props: {
				cwd: CWD,
				entry: makeEntry('manual-only', { disable_model_invocation: true }),
				mobile: false,
				onClose: vi.fn()
			}
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Manual only'));
		expect(screen.getByRole('button', { name: 'Markdown' })).toBeTruthy();
	});

	it('places the resource picker in the detail header', async () => {
		mockRead(() =>
			jsonResponse(
				baseResult({
					resources: {
						paths: ['assets/template.txt', 'references/guide.md', 'scripts/check.py', 'notes.txt'],
						truncated: false
					}
				})
			)
		);

		const screen = await render(SkillDetail, {
			props: { cwd: undefined, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		const header = screen.getByTestId('skill-detail-header').element();
		const body = screen.getByTestId('skill-detail-body').element();
		const picker = screen.getByTestId('skill-resource-picker-trigger').element();

		expect(header.contains(picker)).toBe(true);
		expect(body.contains(picker)).toBe(false);
		expect(picker.textContent).toContain('SKILL.md');
		expect(picker.textContent).toContain('(4)');
	});

	it('sends exactly { name } with the selected CWD header and never a path', async () => {
		const readCalls: RequestInit[] = [];

		mockRead((init) => {
			readCalls.push(init);

			return jsonResponse(baseResult());
		});

		await render(SkillDetail, {
			props: { cwd: CWD, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(readCalls).toHaveLength(1));

		const init = readCalls[0];

		expect(init.method).toBe('POST');
		expect(JSON.parse(init.body as string)).toEqual({ name: 'demo-skill' });
		expect(new Headers(init.headers).get('x-skill-cwd')).toBe(CWD);
	});

	it('creates no database message and no activation record for a preview read', async () => {
		const createBranch = vi.spyOn(DatabaseService, 'createMessageBranch');
		const createBranchPair = vi.spyOn(DatabaseService, 'createMessageBranchPair');
		const recordActivation = vi.spyOn(skillActivationStore, 'recordActivation');

		mockRead(() => jsonResponse(baseResult()));

		const screen = await render(SkillDetail, {
			props: { cwd: CWD, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));
		await screen.getByRole('button', { name: 'Raw' }).click();

		expect(createBranch).not.toHaveBeenCalled();
		expect(createBranchPair).not.toHaveBeenCalled();
		expect(recordActivation).not.toHaveBeenCalled();
		expect(skillActivationStore.isActivated('conv-preview', 'opaque-demo-skill')).toBe(false);
	});

	it('treats a resource read result as an error instead of rendering it', async () => {
		mockRead(() => jsonResponse(resourceResult()));

		const screen = await render(SkillDetail, {
			props: { cwd: undefined, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Could not load the skill'));

		expect(bodyText()).not.toContain('Rendered heading');
		expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy();
	});

	it('keeps the selected name visible and retries the same name and CWD after a failure', async () => {
		const readCalls: RequestInit[] = [];

		let failNext = true;

		mockRead((init) => {
			readCalls.push(init);

			if (failNext) {
				failNext = false;

				return jsonResponse({ error: { code: 500, message: 'boom' } }, 500);
			}

			return jsonResponse(baseResult());
		});

		const screen = await render(SkillDetail, {
			props: { cwd: CWD, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Could not load the skill'));
		expect(bodyText()).toContain('demo-skill');

		await screen.getByRole('button', { name: 'Retry' }).click();

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		expect(readCalls).toHaveLength(2);
		for (const init of readCalls) {
			expect(JSON.parse(init.body as string)).toEqual({ name: 'demo-skill' });
			expect(new Headers(init.headers).get('x-skill-cwd')).toBe(CWD);
		}
	});

	it('aborts the superseded read when the entry changes before the first read resolves', async () => {
		const readA = deferredRead();
		const readB = deferredRead();
		const reads = [readA, readB];

		let index = 0;

		mockRead((init) => {
			const current = reads[Math.min(index++, reads.length - 1)];

			current.attach(init);

			return current.promise;
		});

		const { rerender } = await render(SkillDetail, {
			props: { cwd: undefined, entry: makeEntry('skill-a'), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(readA.signal).toBeDefined());
		expect(readA.signal?.aborted).toBe(false);

		await rerender({
			cwd: undefined,
			entry: makeEntry('skill-b'),
			mobile: false,
			onClose: vi.fn()
		});

		await vi.waitFor(() => expect(readA.signal?.aborted).toBe(true));
		expect(readB.signal).toBeDefined();

		readB.resolve(
			jsonResponse(baseResult({ body_markdown: '# Content of B', source: '---\n# Content of B' }))
		);

		await vi.waitFor(() => expect(bodyText()).toContain('Content of B'));
		expect(bodyText()).not.toContain('Content of A');
	});

	it('never renders a stale response that resolves after its read was superseded', async () => {
		const resolvers: Array<(response: Response) => void> = [];

		// This mock ignores abort; the stale resolution must still be dropped.
		mockRead(() => {
			const { promise, resolve } = Promise.withResolvers<Response>();

			resolvers.push(resolve);

			return promise;
		});

		const { rerender } = await render(SkillDetail, {
			props: { cwd: undefined, entry: makeEntry('skill-a'), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(resolvers).toHaveLength(1));

		await rerender({
			cwd: undefined,
			entry: makeEntry('skill-b'),
			mobile: false,
			onClose: vi.fn()
		});

		await vi.waitFor(() => expect(resolvers).toHaveLength(2));

		resolvers[1](jsonResponse(baseResult({ body_markdown: '# Content of B' })));
		await vi.waitFor(() => expect(bodyText()).toContain('Content of B'));

		resolvers[0](jsonResponse(baseResult({ body_markdown: '# Content of A' })));
		for (let i = 0; i < 25; i++) await Promise.resolve();

		expect(bodyText()).not.toContain('Content of A');
	});

	it('aborts the read and never renders the old detail when the CWD changes', async () => {
		const readOld = deferredRead();
		const readNew = deferredRead();
		const reads = [readOld, readNew];

		let index = 0;

		mockRead((init) => {
			const current = reads[Math.min(index++, reads.length - 1)];

			current.attach(init);

			return current.promise;
		});

		const { rerender } = await render(SkillDetail, {
			props: { cwd: '/srv/old', entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(readOld.signal).toBeDefined());

		await rerender({
			cwd: '/srv/new',
			entry: makeEntry(),
			mobile: false,
			onClose: vi.fn()
		});

		await vi.waitFor(() => expect(readOld.signal?.aborted).toBe(true));
		expect(readNew.signal).toBeDefined();

		readNew.resolve(jsonResponse(baseResult()));

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));
		expect(readOld.signal?.aborted).toBe(true);
	});

	it('aborts the in-flight read when the detail unmounts', async () => {
		const read = deferredRead();

		mockRead((init) => {
			read.attach(init);

			return read.promise;
		});

		const { unmount } = await render(SkillDetail, {
			props: { cwd: undefined, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(read.signal).toBeDefined());
		expect(read.signal?.aborted).toBe(false);

		await unmount();

		expect(read.signal?.aborted).toBe(true);
	});

	it('defaults each newly selected skill back to the markdown mode', async () => {
		mockRead(() => jsonResponse(baseResult()));

		const screen = await render(SkillDetail, {
			props: { cwd: undefined, entry: makeEntry('skill-a'), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		await screen.getByRole('button', { name: 'Raw' }).click();
		expect(screen.getByTestId('skill-detail-raw').query()).not.toBeNull();

		const { rerender } = screen;

		await rerender({
			cwd: undefined,
			entry: makeEntry('skill-b'),
			mobile: false,
			onClose: vi.fn()
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		expect(screen.getByTestId('skill-detail-raw').query()).toBeNull();
		expect(screen.getByTestId('skill-detail-markdown').query()).not.toBeNull();
	});

	it('omits read-result structured metadata from the preview', async () => {
		mockRead(() => jsonResponse(baseResult()));

		const screen = await render(SkillDetail, {
			props: { cwd: undefined, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		// Read-result metadata is intentionally never rendered.
		expect(screen.getByTestId('skill-detail-metadata').query()).toBeNull();
		expect(bodyText()).not.toContain('Structured metadata description');
		expect(bodyText()).not.toContain('MIT');
	});

	it('opens a resource tree and previews the selected supported resource', async () => {
		const readCalls: RequestInit[] = [];
		mockRead((init) => {
			readCalls.push(init);
			const request = JSON.parse(init.body as string) as { path?: string };

			if (request.path === 'references/guide.md') {
				return jsonResponse({
					...resourceResult(),
					resource: { path: 'references/guide.md' },
					source: '# Guide\n\nResource body.'
				});
			}

			return jsonResponse(
				baseResult({
					resources: { paths: ['references/guide.md', 'scripts/check.py'], truncated: false }
				})
			);
		});

		const screen = await render(SkillDetail, {
			props: { cwd: CWD, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));
		await screen.getByTestId('skill-resource-picker-trigger').click();
		await vi.waitFor(() => expect(bodyText()).toContain('references'));
		await screen.getByRole('treeitem', { name: /guide\.md/i }).click();
		await vi.waitFor(() => expect(bodyText()).toContain('Resource body.'));

		expect(readCalls).toHaveLength(2);
		expect(JSON.parse(readCalls[1].body as string)).toEqual({
			name: 'demo-skill',
			path: 'references/guide.md'
		});
	});

	it('keeps the picker closed when the skill exposes no resources', async () => {
		mockRead(() => jsonResponse(baseResult({ resources: { paths: [], truncated: false } })));

		const screen = await render(SkillDetail, {
			props: { cwd: CWD, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));
		await screen.getByTestId('skill-resource-picker-trigger').click();

		expect(screen.getByTestId('skill-resource-picker-tree').query()).toBeNull();
	});
});
