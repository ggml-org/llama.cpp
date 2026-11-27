import {
	HF_STORAGE_KEY_OAUTH,
	HF_STORAGE_KEY_SEPARATOR,
	HF_STORAGE_KEY_TOKEN
} from '@ggml-org/llama-huggingface';
import { STORAGE_APP_NAME } from '#lib/constants/index.ts';
import { huggingFace } from '#lib/stores/huggingface.svelte.ts';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// the sign-in itself is tested in ui/lib/huggingface; these cover what the account does around
// it, as Llama Chat sets it up

const HF_OAUTH_LOCALSTORAGE_KEY = [STORAGE_APP_NAME, HF_STORAGE_KEY_OAUTH].join(
	HF_STORAGE_KEY_SEPARATOR
);
const HF_TOKEN_LOCALSTORAGE_KEY = [STORAGE_APP_NAME, HF_STORAGE_KEY_TOKEN].join(
	HF_STORAGE_KEY_SEPARATOR
);

interface HubMock {
	/** The datasets the account lists; one of them by default. */
	datasets?: { id: string; private: boolean }[];
	/** Status a share's preupload answers with, instead of accepting the file. */
	preupload?: number;
	register?: () => Response;
	whoami?: () => Response;
}

// the Hub's lookup of one dataset: /api/datasets/<owner>/<name>
const DATASET_LOOKUP_URL = /\/api\/datasets\/[^/]+\/[^/?]+$/;

function json(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), { status });
}

function mockHub({ datasets, preupload, register, whoami }: HubMock = {}) {
	vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
		const url = String(input instanceof Request ? input.url : input);

		if (url.endsWith('/oauth/register')) {
			return register ? register() : json({ client_id: 'client-1' }, 201);
		}

		if (url.endsWith('/oauth/token')) {
			return json({ access_token: 'hf_oauth_a', expires_in: 3600, refresh_token: 'r1' });
		}

		if (url.endsWith('/api/whoami-v2')) return whoami ? whoami() : json({ name: 'me' });

		if (url.includes('/api/datasets?')) {
			return json(datasets ?? [{ id: 'me/traces', private: false }]);
		}

		// a share to a dataset that exists: look it up, check the file, commit it
		if (DATASET_LOOKUP_URL.test(url)) return json({});

		if (url.endsWith('/preupload/main')) {
			return preupload
				? json({ error: 'Forbidden' }, preupload)
				: json({ files: [{ uploadMode: 'regular' }] });
		}

		if (url.endsWith('/commit/main')) return json({});

		throw new Error(`unexpected request: ${url}`);
	});
}

function fakePopup() {
	return { close: vi.fn(), location: { href: '' } };
}

/** Signs in through a popup, up to Hugging Face sending the browser back. */
async function signInWithHuggingFace() {
	const popup = fakePopup();

	await huggingFace.signInWithHuggingFace(popup as unknown as Window);

	const authorization = new URL(popup.location.href);
	const callback = new URL(authorization.searchParams.get('redirect_uri') ?? '');

	callback.searchParams.set('code', 'code-1');
	callback.searchParams.set('state', authorization.searchParams.get('state') ?? '');

	return huggingFace.completeSignInFromLocation(callback);
}

beforeEach(() => {
	localStorage.removeItem(HF_OAUTH_LOCALSTORAGE_KEY);
	huggingFace.signOut();
	huggingFace.oauthError = null;
	// the callback clears its parameters from the address; leave the test page's own alone
	vi.spyOn(window.history, 'replaceState').mockImplementation(() => {});
});

afterEach(() => {
	huggingFace.cancelSignInWithHf();
	vi.restoreAllMocks();
	// other tests share this browser's storage: leave no sign-in behind
	huggingFace.signOut();
});

describe('the Hugging Face account of Llama Chat', () => {
	it('signs in with Hugging Face, then checks who signed in and lists their datasets', async () => {
		mockHub();

		const outcome = await signInWithHuggingFace();

		expect(outcome).toEqual({ closeWindow: true, result: { ok: true } });
		await vi.waitFor(() => expect(huggingFace.datasets).toHaveLength(1));
		expect(huggingFace.user?.name).toBe('me');
		expect(huggingFace.isSigningInWithHf).toBe(false);
	});

	it('says when the datasets of the signed-in account are listed, so a share can wait for them', async () => {
		mockHub();
		expect(huggingFace.hasListedDatasets).toBe(false);

		await signInWithHuggingFace();
		await vi.waitFor(() => expect(huggingFace.hasListedDatasets).toBe(true));
		expect(huggingFace.datasets).toHaveLength(1);

		huggingFace.signOut();

		expect(huggingFace.hasListedDatasets).toBe(false);
	});

	it('points back to token sign-in when Hugging Face will not register the app', async () => {
		mockHub({ register: () => json({ error: 'invalid_redirect_uri' }, 400) });

		await huggingFace.signInWithHuggingFace(fakePopup() as unknown as Window);

		expect(huggingFace.isSigningInWithHf).toBe(false);
		expect(huggingFace.oauthError).toContain('does not allow signing in from');
		expect(huggingFace.oauthError).toContain('Sign in with an access token instead.');
	});

	it('signs in with a pasted token and refuses one the Hub rejects', async () => {
		mockHub({ whoami: () => json({ error: 'Invalid credentials' }, 401) });
		await expect(huggingFace.signIn('hf_bad')).rejects.toThrow('did not accept this token');
		expect(localStorage.getItem(HF_TOKEN_LOCALSTORAGE_KEY)).toBeNull();

		vi.restoreAllMocks();
		mockHub();
		await huggingFace.signIn(' hf_good ');

		expect(huggingFace.user?.name).toBe('me');
		expect(localStorage.getItem(HF_TOKEN_LOCALSTORAGE_KEY)).toBe('hf_good');
	});

	it('keeps a remembered token when the Hub cannot be reached, forgets one it rejects', async () => {
		mockHub();
		await huggingFace.signIn('hf_good');
		huggingFace.user = null;

		vi.restoreAllMocks();
		mockHub({ whoami: () => json({ error: 'Down' }, 503) });
		await huggingFace.restoreSession();

		expect(localStorage.getItem(HF_TOKEN_LOCALSTORAGE_KEY)).toBe('hf_good');

		vi.restoreAllMocks();
		mockHub({ whoami: () => json({ error: 'Invalid credentials' }, 401) });
		await huggingFace.restoreSession();

		expect(localStorage.getItem(HF_TOKEN_LOCALSTORAGE_KEY)).toBeNull();
	});

	it('forgets the account and its last dataset on sign-out', async () => {
		mockHub();
		await signInWithHuggingFace();
		await vi.waitFor(() => expect(huggingFace.user?.name).toBe('me'));
		huggingFace.lastDataset = 'me/traces';
		huggingFace.signOut();

		expect(huggingFace.user).toBeNull();
		expect(huggingFace.suggestedDataset).toBe('');
		expect(
			JSON.parse(localStorage.getItem(HF_OAUTH_LOCALSTORAGE_KEY) ?? '{}').tokens
		).toBeUndefined();
	});

	describe('the datasets a share can go to', () => {
		const TRACE = { commitMessage: 'Add trace', content: '{}', isPrivate: false, path: 't.jsonl' };

		// the account remembers what each account shared to, so every test signs in as another one
		let owner = '';
		let datasets: { id: string; private: boolean }[] = [];
		let accountNumber = 0;

		const writableIds = () => huggingFace.writableDatasets.map((dataset) => dataset.id);
		const whoamiAs = (role?: string) => () =>
			json({ auth: role ? { accessToken: { role } } : undefined, name: owner });

		beforeEach(() => {
			owner = `account-${accountNumber++}`;
			datasets = [
				{ id: `${owner}/notes`, private: true },
				{ id: `${owner}/traces`, private: false },
				{ id: `${owner}/images`, private: false }
			];
		});

		it('offers none to a sign-in with Hugging Face, which has shared to none yet', async () => {
			mockHub({ datasets, whoami: whoamiAs() });
			await signInWithHuggingFace();
			await vi.waitFor(() => expect(huggingFace.datasets).toHaveLength(3));

			expect(huggingFace.writableDatasets).toEqual([]);
		});

		it('offers the ones it shared to, the last one first, and remembers them', async () => {
			mockHub({ datasets, whoami: whoamiAs() });
			await signInWithHuggingFace();
			await vi.waitFor(() => expect(huggingFace.datasets).toHaveLength(3));

			await huggingFace.uploadTrace({ ...TRACE, repoId: `${owner}/images` });
			await huggingFace.uploadTrace({ ...TRACE, repoId: `${owner}/notes` });

			expect(writableIds()).toEqual([`${owner}/notes`, `${owner}/images`]);

			// the same account signing in again still has them
			huggingFace.signOut();
			await signInWithHuggingFace();
			await vi.waitFor(() => expect(huggingFace.datasets).toHaveLength(3));

			expect(writableIds().sort()).toEqual([`${owner}/images`, `${owner}/notes`]);
		});

		it('stops offering one the Hub turns down', async () => {
			mockHub({ datasets, whoami: whoamiAs() });
			await signInWithHuggingFace();
			await vi.waitFor(() => expect(huggingFace.datasets).toHaveLength(3));
			await huggingFace.uploadTrace({ ...TRACE, repoId: `${owner}/images` });

			vi.restoreAllMocks();
			mockHub({ datasets, preupload: 403, whoami: whoamiAs() });

			await expect(
				huggingFace.uploadTrace({ ...TRACE, repoId: `${owner}/images` })
			).rejects.toThrow(`cannot add traces to ${owner}/images`);
			expect(huggingFace.writableDatasets).toEqual([]);
		});

		it('offers every one to a pasted token, and none to a token that only reads', async () => {
			mockHub({ datasets, whoami: whoamiAs('write') });
			await huggingFace.signIn('hf_good');
			await vi.waitFor(() => expect(huggingFace.datasets).toHaveLength(3));

			expect(huggingFace.writableDatasets).toHaveLength(3);

			huggingFace.signOut();
			vi.restoreAllMocks();
			mockHub({ datasets, whoami: whoamiAs('read') });
			await huggingFace.signIn('hf_read');
			await vi.waitFor(() => expect(huggingFace.datasets).toHaveLength(3));

			expect(huggingFace.isReadOnly).toBe(true);
			expect(huggingFace.writableDatasets).toEqual([]);
		});

		it('starts a new name with the usual dataset, unless it exists', async () => {
			mockHub({ datasets, whoami: whoamiAs('write') });
			await huggingFace.signIn('hf_good');
			await vi.waitFor(() => expect(huggingFace.datasets).toHaveLength(3));

			expect(huggingFace.newDatasetStart).toBe(`${owner}/llama-chat-traces`);

			huggingFace.datasets = [
				...huggingFace.datasets,
				{ id: `${owner}/llama-chat-traces`, isPrivate: false }
			];

			expect(huggingFace.newDatasetStart).toBe(`${owner}/`);
		});
	});
});
