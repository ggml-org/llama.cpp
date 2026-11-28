// A stand-in for the Hugging Face account that the share dialog reads and calls, as plain state:
// it calls out of nowhere, so a test or a story can set up who is signed in and which datasets
// the account can add to, and watch what the dialog does with them.

import { type HfDataset, HfNotWritableError } from '@ggml-org/llama-huggingface';
import type { HfAccount } from '@ggml-org/llama-huggingface/svelte';
import { SvelteSet } from 'svelte/reactivity';

export const FAKE_OWNER = 'me';
export const FAKE_SHARED_URL = 'https://huggingface.co/datasets/me/a/blob/main/t.jsonl';

/** The part of an account the dialog reads and calls, as plain state. */
export class FakeAccount {
	datasets = $state<HfDataset[]>([]);
	hasListedDatasets = $state(true);
	isLoadingDatasets = $state(false);
	isReadOnly = false;
	isSigningInWithHf = $state(false);
	lastDataset = $state<string | null>(null);
	loadDatasets = async () => {};
	newDatasetStart = `${FAKE_OWNER}/llama-chat-traces`;
	oauthError = $state<string | null>(null);
	/** Repo IDs the Hub turns down. */
	refused = new SvelteSet<string>();
	restoreSession = async () => {};
	signOut = () => {
		this.user = null;
	};

	suggestedDataset = '';

	uploads: string[] = [];
	uploadTrace = async ({ repoId }: { repoId: string }): Promise<string> => {
		this.uploads.push(repoId);

		if (this.refused.has(repoId)) {
			// as the real account does: the dataset is not one it can add to any more
			this.datasets = this.datasets.filter((dataset) => dataset.id !== repoId);

			throw new HfNotWritableError(repoId);
		}

		return FAKE_SHARED_URL;
	};
	user = $state<{ name: string } | null>({ name: FAKE_OWNER });

	get writableDatasets(): HfDataset[] {
		return this.datasets;
	}
}

/** An account that can add to datasets of the given names, as the account `me`. */
export function fakeAccountWith(...names: string[]): FakeAccount {
	const account = new FakeAccount();

	account.datasets = names.map((name) => ({ id: `${FAKE_OWNER}/${name}`, isPrivate: false }));

	return account;
}

export function asHfAccount(account: FakeAccount): HfAccount {
	return account as unknown as HfAccount;
}
