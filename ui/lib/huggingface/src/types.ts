import type { HfOAuthFlowMode, HfTokenRole } from './enums.js';

/** Tokens a sign-in got, as kept in the browser. */
export interface HfOAuthTokens {
	accessToken: string;
	refreshToken?: string;
	/** Epoch milliseconds; unknown when Hugging Face did not say. */
	expiresAt?: number;
	/** The scopes Hugging Face granted, space-separated. */
	scope?: string;
}

/** A sign-in waiting for its callback. */
export interface HfOAuthPendingFlow {
	codeVerifier: string;
	mode: HfOAuthFlowMode;
	/** Where a redirect sign-in returns to. */
	returnUrl: string;
	startedAt: number;
	state: string;
}

/** What the browser keeps: the client as it was registered, and its tokens. */
export interface HfOAuthRecord {
	clientId: string;
	redirectUrl: string;
	scope: string;
	logoUrl?: string;
	pending?: HfOAuthPendingFlow;
	tokens?: HfOAuthTokens;
}

/** How a sign-in ended, as every tab of the app hears it. */
export interface HfOAuthResult {
	ok: boolean;
	error?: string;
}

/** What a page load that came back from the authorization page did. */
export interface HfOAuthCallbackOutcome {
	/** The window was a sign-in popup: it is done and can close. */
	closeWindow: boolean;
	/** Null for a callback this browser did not start, or already finished. */
	result: HfOAuthResult | null;
}

/** What the token endpoint answers, before it is checked: any field may be missing. */
export interface HfTokenResponse {
	access_token?: unknown;
	expires_in?: unknown;
	refresh_token?: unknown;
	scope?: unknown;
}

/** What the registration endpoint answers, before it is checked. */
export interface HfRegistrationResponse {
	client_id?: unknown;
}

export interface HfOAuthClient {
	clientName: string;
	redirectUrl: string;
	scopes: readonly string[];
	logoUrl?: string;
}

export interface HfOAuthSignInOptions {
	/** The name shown on the Hugging Face authorization page. */
	clientName: string;
	/** A public image Hugging Face shows for the app, e.g. among the user's connected apps. */
	logoUrl?: string;
	scopes: readonly string[];
	/** A scope the app cannot do without: tokens that lack it are refused. */
	requiredScope?: string;
	/** The page of the app Hugging Face sends the browser back to, usually its root. */
	callbackUrl: () => string;
	/** localStorage key of the record; also names the channel the tabs share. */
	storageKey: string;
	/** Called in every tab of the app when a sign-in ends. */
	onResult?: (result: HfOAuthResult) => void;
}

// Hub

/** The signed-in account, from `/api/whoami-v2`. */
export interface HfUser {
	name: string;
	fullname?: string;
	/** Set for a user access token; a read token cannot write. */
	tokenRole?: HfTokenRole;
}

/** A dataset of the account. */
export interface HfDataset {
	/** `owner/name` */
	id: string;
	isPrivate: boolean;
}

/** A message of the shared conversation, as the faded preview of a finished share shows it. */
export interface HfTracePreviewLine {
	role: 'user' | 'assistant';
	text: string;
}

/** A trace as the app hands it over: one JSONL file and how its commit reads. */
export interface HfTraceFile {
	path: string;
	content: string;
	commitMessage: string;
	/** The first messages of the conversation, shown faded once it is shared. Nothing is uploaded from it. */
	preview?: HfTracePreviewLine[];
}

/** A trace file and the dataset it goes to. */
export interface HfTraceUpload extends HfTraceFile {
	repoId: string;
	/** Visibility of the dataset when the upload creates it; an existing one keeps its own. */
	isPrivate: boolean;
}

// Account (./svelte)

/** What the app tells the account about itself. */
export interface HfAccountOptions extends Omit<HfOAuthSignInOptions, 'onResult' | 'storageKey'> {
	/** Prefix of the localStorage keys the account keeps its sign-in under, e.g. the app's name. */
	storagePrefix: string;
	/** The dataset name a first share suggests, under the account's own name. */
	defaultDatasetName: string;
}

/** One part of a conversation that can be shared, e.g. all of it or up to a message. */
export interface HfShareScope {
	id: string;
	label: string;
	/** Shown after the label, e.g. `12 messages`. */
	detail?: string;
	/** Offers to include attachments, which are left out otherwise. */
	hasAttachments?: boolean;
	/** Nothing to share yet, e.g. while its messages load. */
	isEmpty?: boolean;
}
