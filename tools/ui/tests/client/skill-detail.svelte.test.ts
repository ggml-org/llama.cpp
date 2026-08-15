// Guards the resizable SKILL.md preview pane: rendered body vs raw source
// separation, Markdown/Raw mode toggle, exact { name } request body with the
// selected CWD header, no database message or activation record from a
// preview read, base-result-only acceptance, intentionally omitted
// read-result metadata, collapsible resource discovery groups, and stale /
// CWD-changed / retried / unmounted response suppression.

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

function baseResult(overrides: Partial<SkillBaseReadResult> = {}): SkillBaseReadResult {
	return {
		kind: 'skill',
		skill: {
			id: 'opaque-demo-skill',
			name: 'demo-skill',
			scope: 'project',
			provider: 'agents',
			metadata: {
				name: 'demo-skill',
				description: 'Structured metadata description',
				license: 'MIT'
			}
		},
		resources: { paths: [], truncated: false },
		source: RAW_SOURCE,
		body_markdown: BODY_MARKDOWN,
		content_xml: '<skill_content name="demo-skill">body</skill_content>',
		diagnostics: [],
		...overrides
	};
}

function resourceResult(name = 'demo-skill'): SkillResourceReadResult {
	return {
		kind: 'resource',
		skill: { id: `opaque-${name}`, name, scope: 'project', provider: 'agents' },
		resource: { path: 'refs/DETAILS.md' },
		content_xml: '<skill_resource>data</skill_resource>',
		diagnostics: []
	};
}

/**
 * A controllable /skills/read fetch: records the request signal and, like a
 * real fetch, rejects the in-flight request when that signal aborts.
 */
function deferredRead() {
	const state: { signal?: AbortSignal } = {};
	const { promise, resolve, reject } = Promise.withResolvers<Response>();

	return {
		promise,
		resolve,
		reject,
		attach(init?: RequestInit) {
			state.signal = init?.signal ?? undefined;
			init?.signal?.addEventListener(
				'abort',
				() => reject(new DOMException('The operation was aborted.', 'AbortError')),
				{ once: true }
			);
		},
		get signal() {
			return state.signal;
		}
	};
}

function mockRead(read: (init: RequestInit) => Response | Promise<Response>) {
	vi.mocked(fetch).mockImplementation(async (url, init) => {
		if (String(url).includes(READ_URL)) return read(init as RequestInit);
		return jsonResponse({ skills: [], catalog_instruction_xml: '', diagnostics: [] });
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
			props: { entry: makeEntry(), cwd: undefined, onClose: vi.fn(), mobile: false }
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

	it('keeps preview context outside the independently scrollable source body', async () => {
		mockRead(() =>
			jsonResponse(
				baseResult({
					resources: { paths: ['references/guide.md'], truncated: false }
				})
			)
		);

		const screen = await render(SkillDetail, {
			props: { entry: makeEntry(), cwd: undefined, onClose: vi.fn(), mobile: false }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		const header = screen.getByTestId('skill-detail-header').element();
		const body = screen.getByTestId('skill-detail-body').element();
		const separator = screen.getByTestId('skill-detail-separator').element();

		expect(header.textContent).toContain('demo-skill');
		// Read-result structured metadata is intentionally not rendered at all.
		expect(screen.getByTestId('skill-detail-metadata').query()).toBeNull();
		expect(header.contains(screen.getByTestId('skill-detail-resources').element())).toBe(true);
		expect(header.contains(screen.getByRole('button', { name: 'Markdown' }).element())).toBe(true);
		expect(header.contains(screen.getByRole('button', { name: 'Raw' }).element())).toBe(true);
		expect(body.contains(screen.getByTestId('skill-detail-markdown').element())).toBe(true);
		expect(body.contains(screen.getByTestId('skill-detail-resources').element())).toBe(false);
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
			props: { entry: makeEntry(), cwd: CWD, onClose: vi.fn(), mobile: false }
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
			props: { entry: makeEntry(), cwd: CWD, onClose: vi.fn(), mobile: false }
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
			props: { entry: makeEntry(), cwd: undefined, onClose: vi.fn(), mobile: false }
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
			props: { entry: makeEntry(), cwd: CWD, onClose: vi.fn(), mobile: false }
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
			props: { entry: makeEntry('skill-a'), cwd: undefined, onClose: vi.fn(), mobile: false }
		});

		await vi.waitFor(() => expect(readA.signal).toBeDefined());
		expect(readA.signal?.aborted).toBe(false);

		await rerender({
			entry: makeEntry('skill-b'),
			cwd: undefined,
			onClose: vi.fn(),
			mobile: false
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
		let index = 0;

		// This mock ignores the abort signal entirely: the superseded request
		// still resolves successfully and must still be dropped.
		mockRead(() => {
			const { promise, resolve } = Promise.withResolvers<Response>();

			resolvers.push(resolve);
			return promise;
		});

		const { rerender } = await render(SkillDetail, {
			props: { entry: makeEntry('skill-a'), cwd: undefined, onClose: vi.fn(), mobile: false }
		});

		await vi.waitFor(() => expect(resolvers).toHaveLength(1));

		await rerender({
			entry: makeEntry('skill-b'),
			cwd: undefined,
			onClose: vi.fn(),
			mobile: false
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
			props: { entry: makeEntry(), cwd: '/srv/old', onClose: vi.fn(), mobile: false }
		});

		await vi.waitFor(() => expect(readOld.signal).toBeDefined());

		await rerender({
			entry: makeEntry(),
			cwd: '/srv/new',
			onClose: vi.fn(),
			mobile: false
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
			props: { entry: makeEntry(), cwd: undefined, onClose: vi.fn(), mobile: false }
		});

		await vi.waitFor(() => expect(read.signal).toBeDefined());
		expect(read.signal?.aborted).toBe(false);

		await unmount();

		expect(read.signal?.aborted).toBe(true);
	});

	it('defaults each newly selected skill back to the markdown mode', async () => {
		mockRead(() => jsonResponse(baseResult()));

		const screen = await render(SkillDetail, {
			props: { entry: makeEntry('skill-a'), cwd: undefined, onClose: vi.fn(), mobile: false }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		await screen.getByRole('button', { name: 'Raw' }).click();
		expect(screen.getByTestId('skill-detail-raw').query()).not.toBeNull();

		const { rerender } = screen;

		await rerender({
			entry: makeEntry('skill-b'),
			cwd: undefined,
			onClose: vi.fn(),
			mobile: false
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		expect(screen.getByTestId('skill-detail-raw').query()).toBeNull();
		expect(screen.getByTestId('skill-detail-markdown').query()).not.toBeNull();
	});

	it('omits read-result structured metadata from the preview', async () => {
		mockRead(() => jsonResponse(baseResult()));

		const screen = await render(SkillDetail, {
			props: { entry: makeEntry(), cwd: undefined, onClose: vi.fn(), mobile: false }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		// Read-result metadata (description, license) is intentionally not
		// rendered anywhere in the preview: no metadata block exists and the
		// values never leak into the markdown body.
		expect(screen.getByTestId('skill-detail-metadata').query()).toBeNull();
		expect(bodyText()).not.toContain('Structured metadata description');
		expect(bodyText()).not.toContain('MIT');
	});

	it('renders only populated convention groups in the required order', async () => {
		const readCalls: RequestInit[] = [];
		const createBranch = vi.spyOn(DatabaseService, 'createMessageBranch');
		const createBranchPair = vi.spyOn(DatabaseService, 'createMessageBranchPair');
		const recordActivation = vi.spyOn(skillActivationStore, 'recordActivation');

		mockRead((init) => {
			readCalls.push(init);
			return jsonResponse(
				baseResult({
					resources: {
						paths: [
							'scripts/run.py',
							'references/API.md',
							'assets/template.json',
							'notes/guide.md'
						],
						truncated: false
					}
				})
			);
		});

		const screen = await render(SkillDetail, {
			props: { entry: makeEntry(), cwd: CWD, onClose: vi.fn(), mobile: false }
		});

		await vi.waitFor(() => expect(screen.getByTestId('skill-detail-resources')).toBeTruthy());

		// Bits UI mounts CollapsibleContent on an effect tick, so the group
		// rows appear just after the region root; wait for them explicitly.
		let groupNodes: Element[] = [];
		await vi.waitFor(() => {
			groupNodes = [
				...screen.baseElement.querySelectorAll('[data-testid="skill-resource-group"]')
			];
			expect(groupNodes.length).toBe(4);
		});

		expect(groupNodes.map((node) => node.textContent)).toEqual([
			expect.stringContaining('Scripts'),
			expect.stringContaining('References'),
			expect.stringContaining('Assets'),
			expect.stringContaining('Other files')
		]);
		expect(bodyText()).toContain('scripts/run.py');
		expect(bodyText()).toContain('references/API.md');

		// Discovery stays read-only: exactly one base { name } request, no
		// database message and no activation record.
		expect(readCalls).toHaveLength(1);
		expect(JSON.parse(readCalls[0].body as string)).toEqual({ name: 'demo-skill' });
		expect(createBranch).not.toHaveBeenCalled();
		expect(createBranchPair).not.toHaveBeenCalled();
		expect(recordActivation).not.toHaveBeenCalled();
	});

	it('omits resource discovery entirely for a SKILL.md-only result', async () => {
		mockRead(() => jsonResponse(baseResult({ resources: { paths: [], truncated: false } })));

		const screen = await render(SkillDetail, {
			props: { entry: makeEntry(), cwd: CWD, onClose: vi.fn(), mobile: false }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Rendered heading'));

		expect(screen.baseElement.querySelector('[data-testid="skill-detail-resources"]')).toBeNull();
	});
});
