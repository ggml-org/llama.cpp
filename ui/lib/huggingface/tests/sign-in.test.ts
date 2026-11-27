import { HfOAuthSignIn, type HfOAuthResult } from '../src/index.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const APP_URL = 'http://localhost:8080/';
const STORAGE_KEY = 'test.hfOAuth';
const SCOPE = 'openid profile contribute-repos';

let requests: { body?: string; url: string }[];
let results: HfOAuthResult[];
let storage: Map<string, string>;
let tokenResponse: () => Response;
let registerResponse: () => Response;

function json(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), { status });
}

function createSignIn(logoUrl?: string): HfOAuthSignIn {
	return new HfOAuthSignIn({
		callbackUrl: () => APP_URL,
		clientName: 'Test app',
		logoUrl,
		onResult: (result) => results.push(result),
		requiredScope: 'contribute-repos',
		scopes: SCOPE.split(' '),
		storageKey: STORAGE_KEY
	});
}

function fakePopup() {
	return { close: vi.fn(), location: { href: '' } };
}

function stored() {
	return JSON.parse(storage.get(STORAGE_KEY) ?? '{}');
}

/** Starts a sign-in in a popup and returns the address Hugging Face would send it back to. */
async function start(signIn: HfOAuthSignIn, popup = fakePopup()): Promise<URL> {
	await signIn.start(popup as unknown as Window);

	const callback = new URL(signIn.redirectUrl);

	callback.searchParams.set('code', 'code-1');
	callback.searchParams.set('state', new URL(popup.location.href).searchParams.get('state') ?? '');

	return callback;
}

beforeEach(() => {
	requests = [];
	results = [];
	storage = new Map();
	registerResponse = () => json({ client_id: 'client-1' }, 201);
	tokenResponse = () =>
		json({
			access_token: 'hf_oauth_a',
			expires_in: 3600,
			refresh_token: 'r1',
			scope: SCOPE
		});

	vi.stubGlobal('localStorage', {
		getItem: (key: string) => storage.get(key) ?? null,
		setItem: (key: string, value: string) => void storage.set(key, value)
	});
	vi.stubGlobal('window', {
		history: { replaceState: vi.fn(), state: null },
		location: { assign: vi.fn(), href: APP_URL, replace: vi.fn() }
	});
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init?: RequestInit) => {
			requests.push({ body: init?.body as string | undefined, url });

			return url.endsWith('/oauth/register') ? registerResponse() : tokenResponse();
		})
	);
});

afterEach(() => {
	vi.unstubAllGlobals();
});

describe('HfOAuthSignIn', () => {
	it('signs in through a popup', async () => {
		const signIn = createSignIn();
		const popup = fakePopup();
		const callback = await start(signIn, popup);
		const authorization = new URL(popup.location.href);

		expect(authorization.origin + authorization.pathname).toBe(
			'https://huggingface.co/oauth/authorize'
		);
		expect(authorization.searchParams.get('redirect_uri')).toBe(`${APP_URL}?hf_oauth=1`);

		expect(await signIn.complete(callback)).toEqual({
			closeWindow: true,
			result: { ok: true }
		});
		expect(results).toEqual([{ ok: true }]);
		expect(signIn.accessToken).toBe('hf_oauth_a');
		expect(await signIn.freshToken()).toBe('hf_oauth_a');

		const exchange = new URLSearchParams(requests.at(-1)?.body);

		expect(exchange.get('code')).toBe('code-1');
		expect(exchange.get('code_verifier')).toMatch(/^[\w-]{43}$/);
	});

	it('sends this tab to the authorization page without a popup, then back', async () => {
		const signIn = createSignIn();

		window.location.href = `${APP_URL}#/chat/1`;
		await signIn.start(null);

		const authorization = new URL(vi.mocked(window.location.assign).mock.calls[0][0] as string);
		const callback = new URL(signIn.redirectUrl);

		callback.searchParams.set('code', 'code-1');
		callback.searchParams.set('state', authorization.searchParams.get('state') ?? '');

		expect(await signIn.complete(callback)).toEqual({
			closeWindow: false,
			result: { ok: true }
		});
		expect(window.location.replace).toHaveBeenCalledWith(`${APP_URL}#/chat/1`);
	});

	it('ignores page loads that are not a callback, and callbacks it did not start', async () => {
		const signIn = createSignIn();
		const callback = await start(signIn);

		callback.searchParams.set('state', 'forged');

		expect(await signIn.complete(new URL(APP_URL))).toBeNull();
		expect(await signIn.complete(callback)).toEqual({
			closeWindow: false,
			result: null
		});
		expect(requests.some((request) => request.url.endsWith('/oauth/token'))).toBe(false);
	});

	it('reuses the registered client on the next sign-in', async () => {
		const signIn = createSignIn();

		await signIn.complete(await start(signIn));
		signIn.signOut();
		await start(signIn);

		expect(requests.filter((request) => request.url.endsWith('/oauth/register'))).toHaveLength(1);
		expect(stored().tokens).toBeUndefined();
	});

	it('registers again when the app gets a logo', async () => {
		await start(createSignIn());
		await start(createSignIn('https://example.com/logo.svg'));

		const registrations = requests.filter((request) => request.url.endsWith('/oauth/register'));

		expect(registrations).toHaveLength(2);
		expect(JSON.parse(registrations[1].body ?? '{}').logo_uri).toBe('https://example.com/logo.svg');
	});

	it('says so on a page that is not secure, and registers nothing', async () => {
		vi.stubGlobal('crypto', { getRandomValues: crypto.getRandomValues.bind(crypto) });

		const popup = fakePopup();

		await expect(createSignIn().start(popup as unknown as Window)).rejects.toThrow('secure page');
		expect(popup.close).toHaveBeenCalled();
		expect(requests).toEqual([]);
	});

	it('closes the popup when Hugging Face will not register the app', async () => {
		registerResponse = () => json({ error: 'invalid_redirect_uri' }, 400);

		const popup = fakePopup();

		await expect(createSignIn().start(popup as unknown as Window)).rejects.toThrow(
			'does not allow signing in from'
		);
		expect(popup.close).toHaveBeenCalled();
	});

	it('refuses tokens without the scope the app needs', async () => {
		tokenResponse = () => json({ access_token: 'hf_oauth_a', scope: 'openid profile' });

		const signIn = createSignIn();
		const outcome = await signIn.complete(await start(signIn));

		expect(outcome?.result?.ok).toBe(false);
		expect(signIn.accessToken).toBeNull();
	});

	it('says when the user turned the sign-in down', async () => {
		const signIn = createSignIn();
		const callback = await start(signIn);

		callback.searchParams.delete('code');
		callback.searchParams.set('error', 'access_denied');

		expect((await signIn.complete(callback))?.result).toEqual({
			error: 'The sign-in was cancelled.',
			ok: false
		});
	});

	it('renews an expired token with its refresh token', async () => {
		tokenResponse = () =>
			json({
				access_token: 'hf_oauth_a',
				expires_in: 0,
				refresh_token: 'r1',
				scope: SCOPE
			});

		const signIn = createSignIn();

		await signIn.complete(await start(signIn));
		tokenResponse = () => json({ access_token: 'hf_oauth_b', expires_in: 3600 });

		expect(await signIn.freshToken()).toBe('hf_oauth_b');
		expect(new URLSearchParams(requests.at(-1)?.body).get('grant_type')).toBe('refresh_token');
		expect(stored().tokens).toMatchObject({
			accessToken: 'hf_oauth_b',
			refreshToken: 'r1'
		});
	});

	it('takes the token another tab renewed when its own refresh is refused', async () => {
		tokenResponse = () =>
			json({
				access_token: 'hf_oauth_a',
				expires_in: 0,
				refresh_token: 'r1',
				scope: SCOPE
			});

		const signIn = createSignIn();

		await signIn.complete(await start(signIn));
		tokenResponse = () => {
			const record = stored();

			storage.set(
				STORAGE_KEY,
				JSON.stringify({
					...record,
					tokens: { accessToken: 'hf_oauth_c', refreshToken: 'r2' }
				})
			);

			return json({ error: 'invalid_grant' }, 400);
		};

		expect(await signIn.freshToken()).toBe('hf_oauth_c');
	});

	it('forgets a sign-in whose refresh token Hugging Face turned down for good', async () => {
		tokenResponse = () =>
			json({ access_token: 'hf_oauth_a', expires_in: 0, refresh_token: 'r1', scope: SCOPE });

		const signIn = createSignIn();

		await signIn.complete(await start(signIn));
		tokenResponse = () => json({ error: 'invalid_grant' }, 400);

		expect(await signIn.freshToken()).toBeNull();
		expect(stored().tokens).toBeUndefined();
		// the registered client stays for the next sign-in
		expect(stored().clientId).toBe('client-1');
	});

	it('keeps a sign-in it could not renew because Hugging Face failed', async () => {
		tokenResponse = () =>
			json({ access_token: 'hf_oauth_a', expires_in: 0, refresh_token: 'r1', scope: SCOPE });

		const signIn = createSignIn();

		await signIn.complete(await start(signIn));
		tokenResponse = () => json({ error: 'unavailable' }, 503);

		await expect(signIn.freshToken()).rejects.toThrow('Renewing the sign-in failed');
		expect(stored().tokens).toMatchObject({ accessToken: 'hf_oauth_a', refreshToken: 'r1' });
	});
});
