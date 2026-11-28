/**
 * HfAccount - the Hugging Face account an app is signed in with, as reactive state
 *
 * Holds who is signed in, their datasets and the dataset traces went to last. The token is
 * either one the user pasted or one "Sign in with Hugging Face" got; the components read the
 * same state either way. An app makes one HfAccount and passes it to them.
 */

import {
	HF_REPO_ID_SEPARATOR,
	HF_SHARED_DATASETS_MAX,
	HF_STORAGE_KEY_DATASET,
	HF_STORAGE_KEY_OAUTH,
	HF_STORAGE_KEY_SEPARATOR,
	HF_STORAGE_KEY_SHARED_DATASETS,
	HF_STORAGE_KEY_TOKEN
} from '../constants.js';
import { HfTokenRole } from '../enums.js';
import { HfNotWritableError } from '../errors.js';
import { listDatasets, uploadTrace, whoAmI } from '../hub.js';
import { HF_MESSAGE } from '../messages.js';
import { HfOAuthSignIn } from '../sign-in.js';
import { readJsonStorage, readStorage, writeJsonStorage, writeStorage } from '../storage.js';
import type {
	HfAccountOptions,
	HfDataset,
	HfOAuthCallbackOutcome,
	HfOAuthResult,
	HfTraceUpload,
	HfUser
} from '../types.js';
import { errorMessage } from '../util.js';

/** The datasets each account shared to from this browser, as stored: account name to repo IDs. */
function readSharedDatasets(key: string): Record<string, string[]> {
	const parsed = readJsonStorage(key);

	if (!parsed || typeof parsed !== 'object') return {};

	return Object.fromEntries(
		Object.entries(parsed).filter(
			(entry): entry is [string, string[]] =>
				Array.isArray(entry[1]) && entry[1].every((id) => typeof id === 'string')
		)
	);
}

export class HfAccount {
	/** Datasets of the account a share can go to, most recently changed first. */
	datasets = $state<HfDataset[]>([]);

	/** Set once the datasets were listed for the account that is signed in, so a share can suggest one of them. */
	hasListedDatasets = $state(false);

	/** Set while the token is being checked against the Hub. */
	isCheckingToken = $state(false);

	/** Set while the account's datasets are being listed. */
	isLoadingDatasets = $state(false);

	/** Set while a sign-in with Hugging Face waits for the authorization page. */
	isSigningInWithHf = $state(false);

	/** Dataset the last share went to, `owner/name`. */
	lastDataset = $state<string | null>(null);

	/** Why the last sign-in with Hugging Face failed. */
	oauthError = $state<string | null>(null);

	/** The account the token belongs to, once checked. */
	user = $state<HfUser | null>(null);

	#datasetKey: string;
	#defaultDatasetName: string;
	#oauth: HfOAuthSignIn;
	/** The pasted access token; a sign-in with Hugging Face keeps its own. */
	#pastedToken = $state<string | null>(null);
	/** The datasets each account shared to from this browser, the most recent first. */
	#shared = $state<Record<string, string[]>>({});
	#sharedKey: string;
	#tokenKey: string;

	constructor(options: HfAccountOptions) {
		const { defaultDatasetName, storagePrefix, ...signIn } = options;
		const key = (name: string) => [storagePrefix, name].join(HF_STORAGE_KEY_SEPARATOR);

		this.#datasetKey = key(HF_STORAGE_KEY_DATASET);
		this.#tokenKey = key(HF_STORAGE_KEY_TOKEN);
		this.#sharedKey = key(HF_STORAGE_KEY_SHARED_DATASETS);
		this.#defaultDatasetName = defaultDatasetName;
		this.#shared = readSharedDatasets(this.#sharedKey);
		this.#pastedToken = readStorage(this.#tokenKey);
		this.lastDataset = readStorage(this.#datasetKey);
		this.#oauth = new HfOAuthSignIn({
			...signIn,
			onResult: (result) => this.#settle(result),
			storageKey: key(HF_STORAGE_KEY_OAUTH)
		});
	}

	/** A read-only access token signs in, but cannot share. */
	get isReadOnly(): boolean {
		return this.user?.tokenRole === HfTokenRole.READ;
	}

	/**
	 * What the name of a new dataset starts with: the usual one under the account's name, or only
	 * the account's name when a dataset of that name exists, so the name is left to be typed.
	 */
	get newDatasetStart(): string {
		if (!this.user) return '';

		const owner = `${this.user.name}${HF_REPO_ID_SEPARATOR}`;
		const usual = `${owner}${this.#defaultDatasetName}`;

		return this.datasets.some((dataset) => dataset.id === usual) ? owner : usual;
	}

	/** Dataset a new share suggests: the last one, else one under the account's name. */
	get suggestedDataset(): string {
		if (this.lastDataset) return this.lastDataset;

		return this.user ? [this.user.name, this.#defaultDatasetName].join(HF_REPO_ID_SEPARATOR) : '';
	}

	/**
	 * The datasets a share can go to, the one the last share went to first. An access token that
	 * was pasted reaches all of the account's datasets. A sign-in with Hugging Face reaches only
	 * the ones the app created, which this browser knows as the ones it shared to.
	 */
	get writableDatasets(): HfDataset[] {
		const user = this.user;

		if (!user || this.isReadOnly) return [];

		const shared = this.#shared[user.name] ?? [];
		const writable = this.#pastedToken
			? this.datasets
			: this.datasets.filter(
					(dataset) => dataset.id === this.lastDataset || shared.includes(dataset.id)
				);
		const last = writable.find((dataset) => dataset.id === this.lastDataset);

		return last ? [last, ...writable.filter((dataset) => dataset !== last)] : writable;
	}

	/** Stop waiting for a sign-in the user gave up on, e.g. by closing the popup. */
	cancelSignInWithHf(): void {
		this.isSigningInWithHf = false;
	}

	/** Finish a sign-in with Hugging Face when this page load is its callback; null otherwise. */
	completeSignInFromLocation(location?: URL): Promise<HfOAuthCallbackOutcome | null> {
		return this.#oauth.complete(location);
	}

	/**
	 * The token to call the Hub with: the pasted one, else the sign-in's, renewed when due.
	 * @returns Null when no one is signed in
	 */
	freshToken(): Promise<string | null> {
		return this.#pastedToken ? Promise.resolve(this.#pastedToken) : this.#oauth.freshToken();
	}

	/** Lists the account's datasets. A failed list leaves it empty, so a new one can be named. */
	async loadDatasets(): Promise<void> {
		const user = this.user;

		if (!user || this.isLoadingDatasets) return;

		this.isLoadingDatasets = true;

		try {
			const token = await this.freshToken();
			const datasets = token ? await listDatasets(token, user.name) : [];

			// the account may have changed while the list was on its way
			if (this.user === user) this.datasets = datasets;
		} catch {
			if (this.user === user) this.datasets = [];
		} finally {
			if (this.user === user) this.hasListedDatasets = true;

			this.isLoadingDatasets = false;
		}
	}

	/**
	 * Checks a remembered token once, to know who is signed in. Only a token the Hub rejects is
	 * forgotten: a Hub that cannot be reached leaves it for the next try.
	 */
	async restoreSession(): Promise<void> {
		if (this.user || this.isCheckingToken) return;

		this.isCheckingToken = true;

		try {
			const token = await this.freshToken();

			if (!token) return;

			const user = await whoAmI(token);

			if (user) this.user = user;
			else this.signOut();
		} catch {
			// offline, or the Hub is down
		} finally {
			this.isCheckingToken = false;
		}
	}

	/**
	 * Checks a pasted access token with the Hub and remembers it.
	 * @throws When the Hub rejects the token
	 */
	async signIn(token: string): Promise<void> {
		const trimmed = token.trim();

		this.isCheckingToken = true;

		try {
			const user = await whoAmI(trimmed);

			if (!user) throw new Error(HF_MESSAGE.TOKEN_REJECTED);

			// one way of signing in at a time
			this.#oauth.signOut();
			this.hasListedDatasets = false;
			this.#pastedToken = trimmed;
			writeStorage(this.#tokenKey, trimmed);
			this.user = user;
		} finally {
			this.isCheckingToken = false;
		}

		void this.loadDatasets();
	}

	/**
	 * Start a sign-in with Hugging Face. Call it from the click that asks for it, with the popup
	 * opened in that same click; without one, this tab goes to the authorization page.
	 */
	async signInWithHuggingFace(popup: Window | null): Promise<void> {
		this.oauthError = null;
		this.isSigningInWithHf = true;

		try {
			await this.#oauth.start(popup);
		} catch (error) {
			this.#settle({ error: errorMessage(error), ok: false });
		}
	}

	signOut(): void {
		this.#forgetAccount();
		this.#oauth.signOut();
		// the next account has its own datasets
		this.#rememberDataset(null);
	}

	/**
	 * Commits a trace to a dataset of the account, and remembers the dataset for the next one.
	 * @returns The URL of the trace file on the Hub
	 */
	async uploadTrace(upload: HfTraceUpload): Promise<string> {
		const token = await this.freshToken();

		if (!token) throw new Error(HF_MESSAGE.SIGN_IN_REQUIRED);

		let url: string;

		try {
			url = await uploadTrace(token, upload);
		} catch (error) {
			// the Hub says this dataset is not one the account can write to: stop offering it
			if (error instanceof HfNotWritableError) {
				this.#rememberShared(upload.repoId, false);

				if (this.lastDataset === upload.repoId) this.#rememberDataset(null);
			}

			throw error;
		}

		if (!this.datasets.some((dataset) => dataset.id === upload.repoId)) {
			this.datasets = [{ id: upload.repoId, isPrivate: upload.isPrivate }, ...this.datasets];
		}

		this.#rememberDataset(upload.repoId);
		this.#rememberShared(upload.repoId, true);

		return url;
	}

	#rememberDataset(repoId: string | null): void {
		this.lastDataset = repoId;
		writeStorage(this.#datasetKey, repoId);
	}

	/** Forgets who is signed in, their datasets and the pasted token, but not the sign-in's tokens. */
	#forgetAccount(): void {
		this.user = null;
		this.datasets = [];
		this.hasListedDatasets = false;
		this.#pastedToken = null;
		writeStorage(this.#tokenKey, null);
	}

	/** Remembers that a dataset can be written to by the account that is signed in, or that it cannot. */
	#rememberShared(repoId: string, isWritable: boolean): void {
		const user = this.user;

		if (!user) return;

		const others = (this.#shared[user.name] ?? []).filter((id) => id !== repoId);
		const ids = isWritable ? [repoId, ...others].slice(0, HF_SHARED_DATASETS_MAX) : others;

		this.#shared = { ...this.#shared, [user.name]: ids };
		writeJsonStorage(this.#sharedKey, this.#shared);
	}

	/** A sign-in with Hugging Face ended, in this tab or another one. */
	#settle(result: HfOAuthResult): void {
		this.isSigningInWithHf = false;
		this.oauthError = result.ok ? null : `${result.error} ${HF_MESSAGE.SIGN_IN_FAILED_FALLBACK}`;

		if (!result.ok) return;

		// one way of signing in at a time; the tab that came back stored the tokens
		this.#forgetAccount();
		void this.restoreSession().then(() => this.loadDatasets());
	}
}
