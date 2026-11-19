/**
 * huggingFaceStore - The Hugging Face account a share uploads with
 *
 * Holds a user access token in localStorage, the account it belongs to, and the
 * dataset traces went to last. Signing in with OAuth can replace the token later
 * without changing what the share dialog reads.
 */

import {
	HF_TOKEN_LOCALSTORAGE_KEY,
	HF_TRACES_DATASET_LOCALSTORAGE_KEY,
	HF_TRACES_DEFAULT_DATASET_NAME
} from '#lib/constants/index.ts';
import { ConversationTransferService } from '#lib/services/conversation-transfer.service.ts';
import { HfTracesService } from '#lib/services/hf-traces.service.ts';
// direct imports between stores, not via the barrel, to avoid circular deps
import { conversationsStore } from '#lib/stores/conversations/index.svelte.ts';
import type { HfUser } from '#lib/types/index.ts';
import { buildConversationTrace } from '#lib/utils/index.ts';
import { browser } from '$app/env';

export interface ShareConversationOptions {
	conversationId: string;
	/** Last message of the trace; the whole current branch when omitted. */
	upToMessageId?: string;
	repoId: string;
	isPrivate: boolean;
	includeAttachments: boolean;
}

function readStorage(key: string): string | null {
	if (!browser) return null;

	try {
		return localStorage.getItem(key);
	} catch {
		return null;
	}
}

function writeStorage(key: string, value: string | null): void {
	if (!browser) return;

	try {
		if (value === null) localStorage.removeItem(key);
		else localStorage.setItem(key, value);
	} catch {
		// storage full or blocked; the value lives for this session only
	}
}

class HuggingFaceStore {
	/** Set while the token is being checked against the Hub. */
	isCheckingToken = $state(false);

	/** Dataset the last share went to, `owner/name`. */
	lastDataset = $state<string | null>(readStorage(HF_TRACES_DATASET_LOCALSTORAGE_KEY));

	/** The account the token belongs to, once checked. */
	user = $state<HfUser | null>(null);

	#token = $state<string | null>(readStorage(HF_TOKEN_LOCALSTORAGE_KEY));

	get hasToken(): boolean {
		return !!this.#token;
	}

	get isSignedIn(): boolean {
		return !!this.#token && !!this.user;
	}

	/** Dataset a new share suggests: the last one, else one under the account's name. */
	get suggestedDataset(): string {
		if (this.lastDataset) return this.lastDataset;

		return this.user ? `${this.user.name}/${HF_TRACES_DEFAULT_DATASET_NAME}` : '';
	}

	/** Checks a remembered token once, so the dialog can show who is signed in. */
	async restoreSession(): Promise<void> {
		if (!this.#token || this.user || this.isCheckingToken) return;

		try {
			await this.signIn(this.#token);
		} catch {
			// a revoked or expired token: ask for a new one
			this.signOut();
		}
	}

	/**
	 * Uploads one branch of a conversation to a dataset as a trace.
	 * @returns The URL of the trace file on the Hub
	 */
	async shareConversation(options: ShareConversationOptions): Promise<string> {
		const token = this.#token;

		if (!token) throw new Error('Sign in to Hugging Face first.');

		const [exported] = await conversationsStore.getConversationsForExport([options.conversationId]);

		if (!exported) throw new Error('Conversation not found.');

		const trace = buildConversationTrace(exported, {
			includeAttachments: options.includeAttachments,
			upToMessageId: options.upToMessageId
		});
		const url = await HfTracesService.uploadTrace(token, {
			commitMessage: `Add trace: ${trace.conv.name}`,
			content: ConversationTransferService.serializeSessionToJsonl(trace),
			isPrivate: options.isPrivate,
			path: ConversationTransferService.generateConversationFilename(trace.conv, trace.messages),
			repoId: options.repoId
		});

		this.lastDataset = options.repoId;
		writeStorage(HF_TRACES_DATASET_LOCALSTORAGE_KEY, options.repoId);

		return url;
	}

	/**
	 * Checks a token with the Hub and remembers it.
	 * @throws When the Hub rejects the token
	 */
	async signIn(token: string): Promise<HfUser> {
		const trimmed = token.trim();

		this.isCheckingToken = true;

		try {
			const user = await HfTracesService.whoAmI(trimmed);

			this.#token = trimmed;
			this.user = user;
			writeStorage(HF_TOKEN_LOCALSTORAGE_KEY, trimmed);

			return user;
		} finally {
			this.isCheckingToken = false;
		}
	}

	signOut(): void {
		this.#token = null;
		this.user = null;
		writeStorage(HF_TOKEN_LOCALSTORAGE_KEY, null);
	}
}

export const huggingFaceStore = new HuggingFaceStore();
