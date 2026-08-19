import SkillDetail from '$lib/components/app/skills/SkillDetail.svelte';
import { DatabaseService } from '$lib/services/database.service';
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import { baseResult, jsonResponse, makeEntry, previewResult } from '../fixtures/skills';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

const READ_URL = '/skills/read';
const CWD = '/srv/project-a';

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
		mockRead(() => jsonResponse(previewResult('demo-skill')));

		const screen = await render(SkillDetail, {
			props: { cwd: undefined, entry: makeEntry('demo-skill'), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Content of demo-skill'));

		const markdownPane = screen.getByTestId('skill-detail-markdown').element();

		expect(markdownPane.textContent).toContain('Content of demo-skill');
		expect(markdownPane.textContent).not.toContain('description: raw frontmatter');

		await screen.getByRole('button', { name: 'Raw' }).click();

		const rawPane = screen.getByTestId('skill-detail-raw').element();

		expect(rawPane.textContent).toContain('---');
		expect(rawPane.textContent).toContain('description: raw frontmatter');
	});

	it('sends exactly { name } with the selected CWD header and never a path', async () => {
		const readCalls: RequestInit[] = [];

		mockRead((init) => {
			readCalls.push(init);

			return jsonResponse(baseResult('demo-skill'));
		});

		await render(SkillDetail, {
			props: { cwd: CWD, entry: makeEntry('demo-skill'), mobile: false, onClose: vi.fn() }
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

		mockRead(() => jsonResponse(baseResult('demo-skill')));

		await render(SkillDetail, {
			props: { cwd: CWD, entry: makeEntry('demo-skill'), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Body of demo-skill'));

		expect(createBranch).not.toHaveBeenCalled();
		expect(createBranchPair).not.toHaveBeenCalled();
		expect(recordActivation).not.toHaveBeenCalled();
		expect(skillActivationStore.isActivated('conv-preview', 'opaque-demo-skill')).toBe(false);
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

			return jsonResponse(baseResult('demo-skill'));
		});

		const screen = await render(SkillDetail, {
			props: { cwd: CWD, entry: makeEntry('demo-skill'), mobile: false, onClose: vi.fn() }
		});

		await vi.waitFor(() => expect(bodyText()).toContain('Could not load the skill'));
		expect(bodyText()).toContain('demo-skill');

		await screen.getByRole('button', { name: 'Retry' }).click();

		await vi.waitFor(() => expect(bodyText()).toContain('Body of demo-skill'));

		expect(readCalls).toHaveLength(2);
		for (const init of readCalls) {
			expect(JSON.parse(init.body as string)).toEqual({ name: 'demo-skill' });
			expect(new Headers(init.headers).get('x-skill-cwd')).toBe(CWD);
		}
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

		resolvers[1](jsonResponse(baseResult('skill-b', { body_markdown: '# Content of B' })));
		await vi.waitFor(() => expect(bodyText()).toContain('Content of B'));

		resolvers[0](jsonResponse(baseResult('skill-a', { body_markdown: '# Content of A' })));
		for (let i = 0; i < 25; i++) await Promise.resolve();

		expect(bodyText()).not.toContain('Content of A');
	});
});
