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

function makeEntry(name = 'demo-skill'): SkillCatalogEntry {
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

	it('keeps each resource group in the detail header outside the scrolling body', async () => {
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
		const separator = screen.getByTestId('skill-detail-separator').element();

		for (const group of ['assets', 'references', 'scripts', 'other']) {
			const resourceGroup = screen.getByTestId(`skill-detail-resources-${group}`).element();

			expect(header.contains(resourceGroup)).toBe(true);
			expect(body.contains(resourceGroup)).toBe(false);
		}

		expect(screen.getByTestId('skill-detail-metadata').query()).toBeNull();
		expect(header.contains(screen.getByRole('button', { name: 'Markdown' }).element())).toBe(true);
		expect(header.contains(screen.getByRole('button', { name: 'Raw' }).element())).toBe(true);
		expect(body.contains(screen.getByTestId('skill-detail-markdown').element())).toBe(true);
		expect(separator.classList.contains('border-t')).toBe(true);
		expect(body.classList.contains('overflow-y-auto')).toBe(true);
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

	it('renders each populated resource group closed without a chevron', async () => {
		const readCalls: RequestInit[] = [];
		const createBranch = vi.spyOn(DatabaseService, 'createMessageBranch');
		const createBranchPair = vi.spyOn(DatabaseService, 'createMessageBranchPair');
		const recordActivation = vi.spyOn(skillActivationStore, 'recordActivation');

		mockRead((init) => {
			readCalls.push(init);

			return jsonResponse(
				baseResult({
					resources: {
						paths: ['assets/template.txt', 'references/guide.md', 'scripts/check.py', 'notes.txt'],
						truncated: false
					}
				})
			);
		});

		const screen = await render(SkillDetail, {
			props: { cwd: CWD, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		for (const group of ['assets', 'references', 'scripts', 'other']) {
			const resourceGroup = screen.getByTestId(`skill-detail-resources-${group}`).element();
			const trigger = screen.getByTestId(`skill-detail-resource-trigger-${group}`).element();

			expect(resourceGroup.getAttribute('data-state')).toBe('closed');
			expect(trigger.getAttribute('aria-expanded')).toBe('false');
		}

		expect(screen.baseElement.querySelector('svg.lucide-chevron-down')).toBeNull();
		expect(screen.baseElement.querySelector('[data-testid="skill-detail-resources"]')).toBeNull();
		expect(readCalls).toHaveLength(1);
		expect(JSON.parse(readCalls[0].body as string)).toEqual({ name: 'demo-skill' });
		expect(createBranch).not.toHaveBeenCalled();
		expect(createBranchPair).not.toHaveBeenCalled();
		expect(recordActivation).not.toHaveBeenCalled();
	});

	it('opens resource groups independently', async () => {
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
			props: { cwd: CWD, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		const state = (group: string) =>
			screen.getByTestId(`skill-detail-resources-${group}`).element().getAttribute('data-state');

		await screen.getByTestId('skill-detail-resource-trigger-references').click();
		await vi.waitFor(() => expect(state('references')).toBe('open'));
		expect(state('assets')).toBe('closed');
		expect(state('scripts')).toBe('closed');
		expect(state('other')).toBe('closed');

		await screen.getByTestId('skill-detail-resource-trigger-scripts').click();
		await vi.waitFor(() => expect(state('scripts')).toBe('open'));
		expect(state('references')).toBe('open');
		expect(state('assets')).toBe('closed');
		expect(state('other')).toBe('closed');
	});

	it('resets resource groups when the selected skill changes', async () => {
		mockRead(() =>
			jsonResponse(
				baseResult({
					resources: {
						paths: ['references/guide.md', 'scripts/check.py'],
						truncated: false
					}
				})
			)
		);

		const screen = await render(SkillDetail, {
			props: { cwd: CWD, entry: makeEntry('skill-a'), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));
		await screen.getByTestId('skill-detail-resource-trigger-references').click();
		await vi.waitFor(() =>
			expect(
				screen.getByTestId('skill-detail-resources-references').element().getAttribute('data-state')
			).toBe('open')
		);

		await screen.rerender({
			cwd: CWD,
			entry: makeEntry('skill-b'),
			mobile: false,
			onClose: vi.fn()
		});

		await vi.waitFor(() => expect(bodyText()).toContain('skill-b'));
		await vi.waitFor(() =>
			expect(
				screen.getByTestId('skill-detail-resources-references').element().getAttribute('data-state')
			).toBe('closed')
		);
	});

	it('omits resource discovery entirely for a SKILL.md-only result', async () => {
		mockRead(() => jsonResponse(baseResult({ resources: { paths: [], truncated: false } })));

		const screen = await render(SkillDetail, {
			props: { cwd: CWD, entry: makeEntry(), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		expect(screen.baseElement.querySelector('[data-testid="skill-detail-resources"]')).toBeNull();
	});
});
