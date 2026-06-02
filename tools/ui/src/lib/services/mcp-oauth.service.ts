import { auth, type OAuthClientProvider } from '@modelcontextprotocol/sdk/client/auth.js';
import type {
	OAuthClientInformationMixed,
	OAuthClientMetadata,
	OAuthTokens
} from '@modelcontextprotocol/sdk/shared/auth.js';
import { checkResourceAllowed } from '@modelcontextprotocol/sdk/shared/auth-utils.js';
import {
	DEFAULT_CLIENT_VERSION,
	MCP_CLIENT_NAME,
	MCP_OAUTH_LOCALSTORAGE_KEY,
	MCP_OAUTH_STATUS_LOCALSTORAGE_KEY
} from '$lib/constants';

const OAUTH_CALLBACK_MARKER_PARAM = 'mcp_oauth_callback';
const OAUTH_CODE_PARAM = 'code';
const OAUTH_ERROR_PARAM = 'error';
const OAUTH_STATE_PARAM = 'state';
const OAUTH_STATUS_EVENT = 'mcp-oauth-status';
let interactiveAuthorizationWindow: Window | null = null;

type StoredMcpOAuthSession = {
	serverId: string;
	serverUrl: string;
	redirectUrl: string;
	returnUrl: string;
	state?: string;
	codeVerifier?: string;
	clientInformation?: OAuthClientInformationMixed;
	tokens?: OAuthTokens;
	createdAt: number;
};

type StoredMcpOAuthState = {
	sessions: Record<string, StoredMcpOAuthSession>;
};

export type McpOAuthCallbackResult = {
	serverId: string;
	serverUrl: string;
	returnUrl: string;
};

export type McpOAuthProviderOptions = {
	serverId: string;
	serverUrl: string;
	redirectUrl?: string;
	returnUrl?: string;
};

export type McpOAuthStatus = {
	phase: 'redirecting' | 'complete' | 'error';
	serverId: string;
	serverUrl: string;
	authorizationUrl?: string;
	message?: string;
	timestamp: number;
};

function isStorageAvailable(): boolean {
	return typeof localStorage !== 'undefined';
}

function getDefaultRedirectUrl(): string {
	const location = globalThis.location;
	const url = new URL(location.href);
	url.search = '';
	url.hash = '';
	url.searchParams.set(OAUTH_CALLBACK_MARKER_PARAM, '1');

	return url.href;
}

function getDefaultReturnUrl(): string {
	return globalThis.location.href;
}

function createStateToken(): string {
	if (globalThis.crypto?.randomUUID) {
		return globalThis.crypto.randomUUID();
	}

	const bytes = new Uint8Array(32);
	globalThis.crypto?.getRandomValues(bytes);

	return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

function loadState(): StoredMcpOAuthState {
	if (!isStorageAvailable()) {
		return { sessions: {} };
	}

	try {
		const raw = localStorage.getItem(MCP_OAUTH_LOCALSTORAGE_KEY);
		const parsed = raw ? JSON.parse(raw) : undefined;

		if (parsed && typeof parsed === 'object' && parsed.sessions) {
			return parsed as StoredMcpOAuthState;
		}
	} catch (error) {
		console.warn('[MCP OAuth] Failed to parse stored OAuth state:', error);
	}

	return { sessions: {} };
}

function saveState(state: StoredMcpOAuthState): void {
	if (!isStorageAvailable()) {
		return;
	}

	localStorage.setItem(MCP_OAUTH_LOCALSTORAGE_KEY, JSON.stringify(state));
}

function publishStatus(status: McpOAuthStatus): void {
	if (isStorageAvailable()) {
		localStorage.setItem(MCP_OAUTH_STATUS_LOCALSTORAGE_KEY, JSON.stringify(status));
	}

	globalThis.dispatchEvent?.(new CustomEvent(OAUTH_STATUS_EVENT, { detail: status }));
}

function clearStatus(): void {
	if (isStorageAvailable()) {
		localStorage.removeItem(MCP_OAUTH_STATUS_LOCALSTORAGE_KEY);
	}
}

function readSession(serverId: string): StoredMcpOAuthSession | undefined {
	return loadState().sessions[serverId];
}

function writeSession(serverId: string, update: Partial<StoredMcpOAuthSession>): StoredMcpOAuthSession {
	const state = loadState();
	const previous = state.sessions[serverId];
	const base: StoredMcpOAuthSession =
		previous ??
		{
			serverId,
			serverUrl: update.serverUrl ?? '',
			redirectUrl: update.redirectUrl ?? getDefaultRedirectUrl(),
			returnUrl: update.returnUrl ?? getDefaultReturnUrl(),
			createdAt: Date.now()
		};
	const next: StoredMcpOAuthSession = {
		...base,
		...update,
		serverId
	};

	state.sessions[serverId] = next;
	saveState(state);

	return next;
}

function deleteSession(serverId: string): void {
	const state = loadState();
	delete state.sessions[serverId];
	saveState(state);
}

function findSessionByState(oauthState: string): StoredMcpOAuthSession | undefined {
	return Object.values(loadState().sessions).find((session) => session.state === oauthState);
}

function stripOAuthCallbackParams(url: URL): URL {
	const cleaned = new URL(url.href);
	cleaned.searchParams.delete(OAUTH_CALLBACK_MARKER_PARAM);
	cleaned.searchParams.delete(OAUTH_CODE_PARAM);
	cleaned.searchParams.delete(OAUTH_ERROR_PARAM);
	cleaned.searchParams.delete('error_description');
	cleaned.searchParams.delete(OAUTH_STATE_PARAM);

	return cleaned;
}

/**
 * Browser-local OAuth provider for remote MCP servers.
 *
 * The MCP SDK owns the protocol details. This provider persists the browser
 * session material needed across the authorization redirect.
 */
export class BrowserMcpOAuthProvider implements OAuthClientProvider {
	readonly serverId: string;
	readonly serverUrl: string;
	private readonly _redirectUrl: string;
	private readonly _returnUrl: string;

	constructor({ serverId, serverUrl, redirectUrl, returnUrl }: McpOAuthProviderOptions) {
		this.serverId = serverId;
		this.serverUrl = serverUrl;
		this._redirectUrl = redirectUrl ?? getDefaultRedirectUrl();
		this._returnUrl = returnUrl ?? getDefaultReturnUrl();

		writeSession(serverId, {
			serverId,
			serverUrl,
			redirectUrl: this._redirectUrl,
			returnUrl: this._returnUrl
		});
	}

	get redirectUrl(): string {
		return this._redirectUrl;
	}

	get clientMetadata(): OAuthClientMetadata {
		return {
			redirect_uris: [this._redirectUrl],
			token_endpoint_auth_method: 'none',
			grant_types: ['authorization_code', 'refresh_token'],
			response_types: ['code'],
			client_name: MCP_CLIENT_NAME,
			software_id: MCP_CLIENT_NAME,
			software_version: DEFAULT_CLIENT_VERSION
		};
	}

	async state(): Promise<string> {
		const existing = readSession(this.serverId)?.state;
		if (existing) {
			return existing;
		}

		const state = createStateToken();
		writeSession(this.serverId, { state });

		return state;
	}

	clientInformation(): OAuthClientInformationMixed | undefined {
		return readSession(this.serverId)?.clientInformation;
	}

	saveClientInformation(clientInformation: OAuthClientInformationMixed): void {
		writeSession(this.serverId, { clientInformation });
	}

	tokens(): OAuthTokens | undefined {
		return readSession(this.serverId)?.tokens;
	}

	saveTokens(tokens: OAuthTokens): void {
		writeSession(this.serverId, { tokens });
	}

	redirectToAuthorization(authorizationUrl: URL): void {
		writeSession(this.serverId, {
			serverId: this.serverId,
			serverUrl: this.serverUrl,
			redirectUrl: this._redirectUrl,
			returnUrl: this._returnUrl
		});

		publishStatus({
			phase: 'redirecting',
			serverId: this.serverId,
			serverUrl: this.serverUrl,
			authorizationUrl: authorizationUrl.href,
			timestamp: Date.now()
		});

		if (interactiveAuthorizationWindow && !interactiveAuthorizationWindow.closed) {
			interactiveAuthorizationWindow.location.href = authorizationUrl.href;
			interactiveAuthorizationWindow = null;
		}
	}

	saveCodeVerifier(codeVerifier: string): void {
		writeSession(this.serverId, { codeVerifier });
	}

	codeVerifier(): string {
		const codeVerifier = readSession(this.serverId)?.codeVerifier;
		if (!codeVerifier) {
			throw new Error('Missing OAuth code verifier for MCP server');
		}

		return codeVerifier;
	}

	async validateResourceURL(_serverUrl: string | URL, resource?: string): Promise<URL | undefined> {
		if (!resource) {
			return undefined;
		}

		if (
			!checkResourceAllowed({
				requestedResource: this.serverUrl,
				configuredResource: resource
			})
		) {
			throw new Error(`Protected resource ${resource} does not match MCP server ${this.serverUrl}`);
		}

		return new URL(resource);
	}

	invalidateCredentials(scope: 'all' | 'client' | 'tokens' | 'verifier'): void {
		if (scope === 'all') {
			deleteSession(this.serverId);

			return;
		}

		const session = readSession(this.serverId);
		if (!session) {
			return;
		}

		const update: Partial<StoredMcpOAuthSession> = {};
		if (scope === 'client') update.clientInformation = undefined;
		if (scope === 'tokens') update.tokens = undefined;
		if (scope === 'verifier') update.codeVerifier = undefined;
		writeSession(this.serverId, update);
	}

	static hasCallbackParams(url = new URL(globalThis.location.href)): boolean {
		return (
			url.searchParams.has(OAUTH_STATE_PARAM) &&
			(url.searchParams.has(OAUTH_CODE_PARAM) || url.searchParams.has(OAUTH_ERROR_PARAM))
		);
	}

	static async completeCallbackFromLocation(
		url = new URL(globalThis.location.href)
	): Promise<McpOAuthCallbackResult | null> {
		if (!this.hasCallbackParams(url)) {
			return null;
		}

		const state = url.searchParams.get(OAUTH_STATE_PARAM);
		if (!state) {
			throw new Error('Missing OAuth state parameter');
		}

		const session = findSessionByState(state);
		if (!session) {
			throw new Error('No pending MCP OAuth session matches this callback');
		}

		const error = url.searchParams.get(OAUTH_ERROR_PARAM);
		if (error) {
			publishStatus({
				phase: 'error',
				serverId: session.serverId,
				serverUrl: session.serverUrl,
				message: error,
				timestamp: Date.now()
			});
			throw new Error(`MCP OAuth failed: ${error}`);
		}

		const authorizationCode = url.searchParams.get(OAUTH_CODE_PARAM);
		if (!authorizationCode) {
			throw new Error('Missing OAuth authorization code');
		}

		const provider = new BrowserMcpOAuthProvider({
			serverId: session.serverId,
			serverUrl: session.serverUrl,
			redirectUrl: session.redirectUrl,
			returnUrl: session.returnUrl
		});

		await auth(provider, {
			serverUrl: session.serverUrl,
			authorizationCode
		});
		writeSession(session.serverId, { state: undefined, codeVerifier: undefined });
		publishStatus({
			phase: 'complete',
			serverId: session.serverId,
			serverUrl: session.serverUrl,
			timestamp: Date.now()
		});

		const returnUrl = stripOAuthCallbackParams(new URL(session.returnUrl));
		globalThis.history?.replaceState(null, '', returnUrl.href);
		if (globalThis.opener) {
			setTimeout(() => globalThis.close?.(), 500);
		}

		return {
			serverId: session.serverId,
			serverUrl: session.serverUrl,
			returnUrl: returnUrl.href
		};
	}

	static clearServer(serverId: string): void {
		deleteSession(serverId);
	}

	static clearStatus(): void {
		clearStatus();
	}

	static beginInteractiveAuthorization(targetWindow: Window | null): void {
		interactiveAuthorizationWindow = targetWindow;
	}

	static subscribeStatus(listener: (status: McpOAuthStatus) => void): () => void {
		const onCustomEvent = (event: Event) => {
			listener((event as CustomEvent<McpOAuthStatus>).detail);
		};
		const onStorage = (event: StorageEvent) => {
			if (event.key !== MCP_OAUTH_STATUS_LOCALSTORAGE_KEY || !event.newValue) {
				return;
			}

			try {
				listener(JSON.parse(event.newValue) as McpOAuthStatus);
			} catch (error) {
				console.warn('[MCP OAuth] Failed to parse OAuth status event:', error);
			}
		};

		globalThis.addEventListener?.(OAUTH_STATUS_EVENT, onCustomEvent);
		globalThis.addEventListener?.('storage', onStorage);

		return () => {
			globalThis.removeEventListener?.(OAUTH_STATUS_EVENT, onCustomEvent);
			globalThis.removeEventListener?.('storage', onStorage);
		};
	}
}
