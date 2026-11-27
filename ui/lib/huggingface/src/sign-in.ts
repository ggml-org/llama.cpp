/**
 * A sign-in with Hugging Face, start to finish, in the browser.
 *
 * start() sends a popup (or, without one, this tab) to the authorization page. Hugging Face
 * sends the browser back to the app with a code, and complete(), called on that page load,
 * trades it for tokens. The popup may not be able to reach the tab that opened it (an app
 * served with COOP same-origin), so the result goes out on a BroadcastChannel to every tab.
 * The registered client and the tokens are kept in localStorage.
 */

import {
	authorizationUrl,
	createCodeChallenge,
	createRandomString,
	exchangeCode,
	hasScope,
	refreshTokens,
	registerClient
} from './client.js';
import {
	HF_OAUTH_CALLBACK_PARAM,
	HF_OAUTH_CALLBACK_VALUE,
	HF_OAUTH_EXPIRY_MARGIN_MS,
	HF_OAUTH_PENDING_TTL_MS,
	HF_OAUTH_SCOPE_SEPARATOR
} from './constants.js';
import { HfOAuthErrorCode, HfOAuthFlowMode, HfOAuthParam } from './enums.js';
import { HfTokenRejectedError } from './errors.js';
import { HF_MESSAGE } from './messages.js';
import { readJsonStorage, writeJsonStorage } from './storage.js';
import type {
	HfOAuthCallbackOutcome,
	HfOAuthRecord,
	HfOAuthResult,
	HfOAuthSignInOptions
} from './types.js';
import { errorMessage } from './util.js';

/** The callback's own query parameters, cleared from the address once read. */
const CALLBACK_PARAMS = [
	HF_OAUTH_CALLBACK_PARAM,
	HfOAuthParam.CODE,
	HfOAuthParam.STATE,
	HfOAuthParam.ERROR,
	HfOAuthParam.ERROR_DESCRIPTION
];

export class HfOAuthSignIn {
	#channel: BroadcastChannel | null = null;
	#options: HfOAuthSignInOptions;

	constructor(options: HfOAuthSignInOptions) {
		this.#options = options;

		if (typeof window === 'undefined' || typeof BroadcastChannel === 'undefined') return;

		this.#channel = new BroadcastChannel(options.storageKey);
		this.#channel.onmessage = (event: MessageEvent<HfOAuthResult>) =>
			options.onResult?.(event.data);
	}

	/** The stored access token, which may have expired: freshToken() renews it. */
	get accessToken(): string | null {
		return this.#read()?.tokens?.accessToken ?? null;
	}

	/** The page Hugging Face sends the browser back to: the app's callback page, with a marker. */
	get redirectUrl(): string {
		const url = new URL(this.#options.callbackUrl());

		url.searchParams.set(HF_OAUTH_CALLBACK_PARAM, HF_OAUTH_CALLBACK_VALUE);

		return url.href;
	}

	/**
	 * Finishes a sign-in when this page load is Hugging Face sending the browser back.
	 * Reads and clears the callback's parameters; returns null for any other page load.
	 */
	async complete(
		location: URL = new URL(window.location.href)
	): Promise<HfOAuthCallbackOutcome | null> {
		if (!location.searchParams.has(HF_OAUTH_CALLBACK_PARAM)) return null;

		const params = new URLSearchParams(location.search);
		const cleaned = new URL(location.href);

		for (const name of CALLBACK_PARAMS) cleaned.searchParams.delete(name);

		// a reload must not run the exchange again
		window.history.replaceState(window.history.state, '', cleaned.href);

		const record = this.#read();
		const pending = record?.pending;
		const state = params.get(HfOAuthParam.STATE);

		if (
			!record ||
			!pending ||
			pending.state !== state ||
			Date.now() - pending.startedAt > HF_OAUTH_PENDING_TTL_MS
		) {
			return { closeWindow: false, result: null };
		}

		this.#write({ ...record, pending: undefined });

		const result = await this.#exchange(record, params, pending.codeVerifier);
		const closeWindow = pending.mode === HfOAuthFlowMode.POPUP;

		this.#options.onResult?.(result);
		// a channel does not hear its own messages: the other tabs do
		this.#channel?.postMessage(result);

		// a sign-in that took over this tab goes back to where it started
		if (!closeWindow && pending.returnUrl !== cleaned.href) {
			window.location.replace(pending.returnUrl);
		}

		return { closeWindow, result };
	}

	/**
	 * The access token to call the Hub with, renewed when it is about to expire.
	 * @returns Null when no one is signed in or the sign-in cannot be renewed
	 * @throws When renewing fails
	 */
	async freshToken(): Promise<string | null> {
		const record = this.#read();
		const tokens = record?.tokens;

		if (!record || !tokens) return null;

		if (!tokens.expiresAt || Date.now() < tokens.expiresAt - HF_OAUTH_EXPIRY_MARGIN_MS) {
			return tokens.accessToken;
		}

		if (!tokens.refreshToken) return null;

		try {
			const renewed = await refreshTokens(record.clientId, tokens.refreshToken);

			this.#write({ ...record, tokens: renewed });

			return renewed.accessToken;
		} catch (error) {
			// another tab may have renewed first, which can spend the refresh token
			const current = this.#read()?.tokens;

			if (current && current.accessToken !== tokens.accessToken) return current.accessToken;

			// a refresh token Hugging Face turned down is no use any more: this is not a sign-in
			// that an unreachable Hub only keeps from renewing
			if (error instanceof HfTokenRejectedError) {
				this.signOut();

				return null;
			}

			throw error;
		}
	}

	/** Forgets the tokens. The client registration stays: the next sign-in reuses it. */
	signOut(): void {
		const record = this.#read();

		if (record?.tokens) this.#write({ ...record, tokens: undefined });
	}

	/**
	 * Starts a sign-in. Call it from the click that asks for it, with the popup opened in that
	 * same click (browsers block a popup opened later); without one, this tab goes to the
	 * authorization page.
	 * @throws When Hugging Face will not register the app; the popup is closed
	 */
	async start(popup: Window | null): Promise<void> {
		const { clientName, logoUrl, scopes } = this.#options;
		const redirectUrl = this.redirectUrl;
		const scope = scopes.join(HF_OAUTH_SCOPE_SEPARATOR);
		const record = this.#read();
		// a client is registered with its redirect URL, scopes and logo: a change registers again
		const known =
			record?.redirectUrl === redirectUrl && record.scope === scope && record.logoUrl === logoUrl
				? record
				: null;

		try {
			const codeVerifier = createRandomString();
			// first, so a page that cannot sign in registers nothing
			const codeChallenge = await createCodeChallenge(codeVerifier);
			const clientId =
				known?.clientId ?? (await registerClient({ clientName, logoUrl, redirectUrl, scopes }));
			const state = createRandomString();

			this.#write({
				clientId,
				pending: {
					codeVerifier,
					mode: popup ? HfOAuthFlowMode.POPUP : HfOAuthFlowMode.REDIRECT,
					returnUrl: window.location.href,
					startedAt: Date.now(),
					state
				},
				logoUrl,
				redirectUrl,
				scope,
				tokens: known?.tokens
			});

			const { href } = authorizationUrl({
				clientId,
				codeChallenge,
				redirectUrl,
				scopes,
				state
			});

			if (popup) popup.location.href = href;
			else window.location.assign(href);
		} catch (error) {
			popup?.close();

			throw error;
		}
	}

	async #exchange(
		record: HfOAuthRecord,
		params: URLSearchParams,
		codeVerifier: string
	): Promise<HfOAuthResult> {
		const failure = params.get(HfOAuthParam.ERROR);
		const code = params.get(HfOAuthParam.CODE);
		const { requiredScope } = this.#options;

		if (failure === HfOAuthErrorCode.ACCESS_DENIED) {
			return { error: HF_MESSAGE.SIGN_IN_CANCELLED, ok: false };
		}

		if (failure) return { error: HF_MESSAGE.SIGN_IN_REFUSED, ok: false };

		if (!code) return { error: HF_MESSAGE.CODE_MISSING, ok: false };

		try {
			const { clientId, redirectUrl } = record;
			const tokens = await exchangeCode({
				clientId,
				code,
				codeVerifier,
				redirectUrl
			});

			if (requiredScope && !hasScope(tokens, requiredScope)) {
				return { error: HF_MESSAGE.SCOPE_NOT_GRANTED, ok: false };
			}

			this.#write({ ...record, pending: undefined, tokens });

			return { ok: true };
		} catch (error) {
			return { error: errorMessage(error), ok: false };
		}
	}

	#read(): HfOAuthRecord | null {
		const parsed = readJsonStorage(this.#options.storageKey);

		return parsed && typeof parsed === 'object' ? (parsed as HfOAuthRecord) : null;
	}

	/** A sign-in that cannot be kept (no storage) lasts until the page is closed. */
	#write(record: HfOAuthRecord): void {
		writeJsonStorage(this.#options.storageKey, record);
	}
}
