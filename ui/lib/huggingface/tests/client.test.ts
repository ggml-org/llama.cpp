import {
	authorizationUrl,
	createCodeChallenge,
	createRandomString,
	exchangeCode,
	hasScope,
	refreshTokens,
	registerClient
} from '../src/index.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const REDIRECT = 'http://192.168.1.5:8080/?hf_oauth=1';
const SCOPES = ['openid', 'profile', 'contribute-repos'];
const LOGO = 'https://example.com/logo.svg';
const CLIENT = {
	clientName: 'Test app',
	logoUrl: LOGO,
	redirectUrl: REDIRECT,
	scopes: SCOPES
};

interface Call {
	url: string;
	init?: RequestInit;
}

let calls: Call[];

function respond(handler: (url: string) => Response) {
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init?: RequestInit) => {
			calls.push({ init, url });

			return handler(url);
		})
	);
}

function json(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), { status });
}

beforeEach(() => {
	calls = [];
});

afterEach(() => {
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

describe('OAuth client', () => {
	it('derives the S256 challenge of a verifier (RFC 7636 appendix B)', async () => {
		expect(await createCodeChallenge('dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk')).toBe(
			'E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM'
		);
	});

	it('makes random strings URL-safe and different each time', () => {
		expect(createRandomString()).toMatch(/^[\w-]{43}$/);
		expect(createRandomString()).not.toBe(createRandomString());
	});

	it('registers a public client for the redirect URL with its scopes', async () => {
		respond(() => json({ client_id: 'client-1' }, 201));

		const clientId = await registerClient(CLIENT);
		const body = JSON.parse(calls[0].init?.body as string);

		expect(clientId).toBe('client-1');
		expect(calls[0].url).toBe('https://huggingface.co/oauth/register');
		expect(body).toMatchObject({
			client_name: 'Test app',
			logo_uri: LOGO,
			redirect_uris: [REDIRECT],
			response_types: ['code'],
			scope: 'openid profile contribute-repos',
			token_endpoint_auth_method: 'none'
		});
	});

	it('says which origin Hugging Face would not register', async () => {
		respond(() => json({ error: 'invalid_redirect_uri', error_description: 'not allowed' }, 400));

		await expect(registerClient(CLIENT)).rejects.toThrow(
			'Hugging Face does not allow signing in from http://192.168.1.5:8080 (invalid_redirect_uri: not allowed).'
		);
	});

	it('reports a registration it could not send', async () => {
		vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

		await expect(registerClient(CLIENT)).rejects.toThrow(
			'Could not reach Hugging Face to start the sign-in.'
		);
	});

	it('builds the authorization URL with PKCE and the state', () => {
		const url = authorizationUrl({
			clientId: 'client-1',
			codeChallenge: 'challenge',
			redirectUrl: REDIRECT,
			scopes: SCOPES,
			state: 'state-1'
		});

		expect(url.origin + url.pathname).toBe('https://huggingface.co/oauth/authorize');
		expect(Object.fromEntries(url.searchParams)).toEqual({
			client_id: 'client-1',
			code_challenge: 'challenge',
			code_challenge_method: 'S256',
			redirect_uri: REDIRECT,
			response_type: 'code',
			scope: 'openid profile contribute-repos',
			state: 'state-1'
		});
	});

	it('trades a code for tokens without a client secret', async () => {
		vi.useFakeTimers({ now: 1_000_000 });
		respond(() =>
			json({
				access_token: 'hf_oauth_a',
				expires_in: 3600,
				refresh_token: 'r1',
				scope: 'openid profile contribute-repos',
				token_type: 'bearer'
			})
		);

		const tokens = await exchangeCode({
			clientId: 'client-1',
			code: 'code-1',
			codeVerifier: 'verifier-1',
			redirectUrl: REDIRECT
		});
		const sent = new URLSearchParams(calls[0].init?.body as string);

		expect(calls[0].url).toBe('https://huggingface.co/oauth/token');
		expect(Object.fromEntries(sent)).toEqual({
			client_id: 'client-1',
			code: 'code-1',
			code_verifier: 'verifier-1',
			grant_type: 'authorization_code',
			redirect_uri: REDIRECT
		});
		expect(tokens).toEqual({
			accessToken: 'hf_oauth_a',
			expiresAt: 1_000_000 + 3_600_000,
			refreshToken: 'r1',
			scope: 'openid profile contribute-repos'
		});
	});

	it('passes on why a code was refused', async () => {
		respond(() => json({ error: 'invalid_grant', error_description: 'code expired' }, 400));

		await expect(
			exchangeCode({
				clientId: 'c',
				code: 'x',
				codeVerifier: 'v',
				redirectUrl: REDIRECT
			})
		).rejects.toThrow('The sign-in failed: invalid_grant: code expired.');
	});

	it('keeps the refresh token when a refresh hands out no new one', async () => {
		respond(() => json({ access_token: 'hf_oauth_b', expires_in: 60 }));

		const tokens = await refreshTokens('client-1', 'r1');
		const sent = new URLSearchParams(calls[0].init?.body as string);

		expect(sent.get('grant_type')).toBe('refresh_token');
		expect(sent.get('refresh_token')).toBe('r1');
		expect(tokens).toMatchObject({
			accessToken: 'hf_oauth_b',
			refreshToken: 'r1'
		});
	});

	it('tells whether tokens carry a scope', () => {
		const scope = 'contribute-repos';

		expect(hasScope({ accessToken: 'a', scope: 'openid contribute-repos' }, scope)).toBe(true);
		expect(hasScope({ accessToken: 'a', scope: 'openid profile' }, scope)).toBe(false);
		expect(hasScope({ accessToken: 'a' }, scope)).toBe(true);
	});
});
