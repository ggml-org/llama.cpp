/**
 * The OAuth calls of a public client in the browser: dynamic client registration, then the
 * authorization code flow with PKCE. The endpoints answer cross-origin requests, so no server
 * is involved. Stateless: HfOAuthSignIn (./sign-in.ts) keeps what a sign-in needs between them.
 */

import {
	HF_CONTENT_TYPE_FORM,
	HF_CONTENT_TYPE_JSON,
	HF_HEADER_CONTENT_TYPE,
	HF_HTTP_CLIENT_ERROR_MIN,
	HF_HTTP_POST,
	HF_HTTP_SERVER_ERROR_MIN,
	HF_MS_PER_SECOND,
	HF_OAUTH_AUTHORIZE_URL,
	HF_OAUTH_CODE_CHALLENGE_HASH,
	HF_OAUTH_CODE_CHALLENGE_METHOD,
	HF_OAUTH_RANDOM_BYTES,
	HF_OAUTH_REGISTER_URL,
	HF_OAUTH_RESPONSE_TYPE,
	HF_OAUTH_SCOPE_SEPARATOR,
	HF_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD,
	HF_OAUTH_TOKEN_URL
} from './constants.js';
import { HfOAuthGrantType, HfOAuthParam } from './enums.js';
import { HfTokenRejectedError } from './errors.js';
import { HF_ACTION, type HfAction, HF_MESSAGE } from './messages.js';
import type {
	HfOAuthClient,
	HfOAuthTokens,
	HfRegistrationResponse,
	HfTokenResponse
} from './types.js';
import { bytesToBase64Url } from './util.js';

/** A random URL-safe string: a PKCE verifier, or a state. */
export function createRandomString(): string {
	return bytesToBase64Url(crypto.getRandomValues(new Uint8Array(HF_OAUTH_RANDOM_BYTES)));
}

/** The S256 challenge of a PKCE verifier. */
export async function createCodeChallenge(verifier: string): Promise<string> {
	// the Web Crypto digest exists only on a secure page
	if (!globalThis.crypto?.subtle) throw new Error(HF_MESSAGE.SECURE_PAGE_NEEDED);

	const digest = await crypto.subtle.digest(
		HF_OAUTH_CODE_CHALLENGE_HASH,
		new TextEncoder().encode(verifier)
	);

	return bytesToBase64Url(new Uint8Array(digest));
}

/** The OAuth error of a failed response, as one readable line. */
async function describeError(response: Response): Promise<string> {
	try {
		const body = await response.json();
		const parts = [body?.[HfOAuthParam.ERROR], body?.[HfOAuthParam.ERROR_DESCRIPTION]].filter(
			(part) => typeof part === 'string' && part
		);

		if (parts.length) return parts.join(': ');
	} catch {
		// not JSON; keep the status
	}

	return HF_MESSAGE.status(response.status);
}

/** A POST that says `unreachable` when no answer comes, instead of the browser's own error. */
async function post(
	url: string,
	body: string,
	contentType: string,
	unreachable: string
): Promise<Response> {
	try {
		return await fetch(url, {
			body,
			headers: { [HF_HEADER_CONTENT_TYPE]: contentType },
			method: HF_HTTP_POST
		});
	} catch {
		throw new Error(unreachable);
	}
}

async function requestTokens(
	fields: Record<string, string>,
	action: HfAction
): Promise<HfOAuthTokens> {
	const now = Date.now();
	const response = await post(
		HF_OAUTH_TOKEN_URL,
		new URLSearchParams(fields).toString(),
		HF_CONTENT_TYPE_FORM,
		HF_MESSAGE.tokenRequestUnreachable(action)
	);

	if (!response.ok) {
		const message = HF_MESSAGE.tokenRequestFailed(action, await describeError(response));

		// a 4xx is the endpoint saying no, for good; a 5xx may pass
		throw response.status >= HF_HTTP_CLIENT_ERROR_MIN && response.status < HF_HTTP_SERVER_ERROR_MIN
			? new HfTokenRejectedError(message)
			: new Error(message);
	}

	const body: HfTokenResponse = await response.json();

	if (typeof body?.access_token !== 'string') throw new Error(HF_MESSAGE.ACCESS_TOKEN_MISSING);

	return {
		accessToken: body.access_token,
		expiresAt:
			typeof body.expires_in === 'number' ? now + body.expires_in * HF_MS_PER_SECOND : undefined,
		refreshToken: typeof body.refresh_token === 'string' ? body.refresh_token : undefined,
		scope: typeof body.scope === 'string' ? body.scope : undefined
	};
}

/**
 * Registers the app as a public client for one redirect URL.
 * @returns The client ID
 * @throws When Hugging Face turns the registration down
 */
export async function registerClient(client: HfOAuthClient): Promise<string> {
	const response = await post(
		HF_OAUTH_REGISTER_URL,
		JSON.stringify({
			client_name: client.clientName,
			grant_types: [HfOAuthGrantType.AUTHORIZATION_CODE, HfOAuthGrantType.REFRESH_TOKEN],
			logo_uri: client.logoUrl,
			redirect_uris: [client.redirectUrl],
			response_types: [HF_OAUTH_RESPONSE_TYPE],
			scope: client.scopes.join(HF_OAUTH_SCOPE_SEPARATOR),
			token_endpoint_auth_method: HF_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD
		}),
		HF_CONTENT_TYPE_JSON,
		HF_MESSAGE.APP_NOT_REGISTERED
	);

	if (!response.ok) {
		throw new Error(
			HF_MESSAGE.registrationRefused(
				new URL(client.redirectUrl).origin,
				await describeError(response)
			)
		);
	}

	const body: HfRegistrationResponse = await response.json();

	if (typeof body?.client_id !== 'string') throw new Error(HF_MESSAGE.CLIENT_ID_MISSING);

	return body.client_id;
}

/** The Hugging Face page that asks the user to let the app in. */
export function authorizationUrl(params: {
	clientId: string;
	codeChallenge: string;
	redirectUrl: string;
	scopes: readonly string[];
	state: string;
}): URL {
	const url = new URL(HF_OAUTH_AUTHORIZE_URL);

	url.searchParams.set(HfOAuthParam.CLIENT_ID, params.clientId);
	url.searchParams.set(HfOAuthParam.CODE_CHALLENGE, params.codeChallenge);
	url.searchParams.set(HfOAuthParam.CODE_CHALLENGE_METHOD, HF_OAUTH_CODE_CHALLENGE_METHOD);
	url.searchParams.set(HfOAuthParam.REDIRECT_URI, params.redirectUrl);
	url.searchParams.set(HfOAuthParam.RESPONSE_TYPE, HF_OAUTH_RESPONSE_TYPE);
	url.searchParams.set(HfOAuthParam.SCOPE, params.scopes.join(HF_OAUTH_SCOPE_SEPARATOR));
	url.searchParams.set(HfOAuthParam.STATE, params.state);

	return url;
}

/**
 * Trades the code the authorization page sent back for tokens.
 * @throws When Hugging Face refuses the code
 */
export function exchangeCode(params: {
	clientId: string;
	code: string;
	codeVerifier: string;
	redirectUrl: string;
}): Promise<HfOAuthTokens> {
	return requestTokens(
		{
			[HfOAuthParam.CLIENT_ID]: params.clientId,
			[HfOAuthParam.CODE]: params.code,
			[HfOAuthParam.CODE_VERIFIER]: params.codeVerifier,
			[HfOAuthParam.GRANT_TYPE]: HfOAuthGrantType.AUTHORIZATION_CODE,
			[HfOAuthParam.REDIRECT_URI]: params.redirectUrl
		},
		HF_ACTION.SIGN_IN
	);
}

/**
 * Gets a new access token with a refresh token.
 * @throws HfTokenRejectedError when the refresh token is no longer good; any other error when
 *   Hugging Face could not be reached or failed
 */
export async function refreshTokens(
	clientId: string,
	refreshToken: string
): Promise<HfOAuthTokens> {
	const tokens = await requestTokens(
		{
			[HfOAuthParam.CLIENT_ID]: clientId,
			[HfOAuthParam.GRANT_TYPE]: HfOAuthGrantType.REFRESH_TOKEN,
			[HfOAuthParam.REFRESH_TOKEN]: refreshToken
		},
		HF_ACTION.SIGN_IN_RENEWAL
	);

	// a refresh may hand out no new refresh token: the old one stays good
	return { ...tokens, refreshToken: tokens.refreshToken ?? refreshToken };
}

/** Whether tokens carry a scope; tokens that do not say which they carry are let through. */
export function hasScope(tokens: HfOAuthTokens, scope: string): boolean {
	return !tokens.scope || tokens.scope.split(HF_OAUTH_SCOPE_SEPARATOR).includes(scope);
}
