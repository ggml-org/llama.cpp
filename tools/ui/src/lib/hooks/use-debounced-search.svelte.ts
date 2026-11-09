import { debounce } from '$lib/utils/debounce';

/**
 * Shared debounced async-search machinery for the chat-form pickers:
 * AbortController + sequence counter to discard stale responses, a
 * debounce, and a live `isSearching` flag.
 */

export interface UseDebouncedSearchOptions {
	debounceMs: number;
	/** Fire-time guard: a scheduled call that outlives a reset is dropped. */
	canRun: () => boolean;
	/** Live query, used to drop a scheduled call whose query changed. */
	getQuery: () => string;
	/** Perform the search and commit results; bail out when `isCurrent()` is false. */
	run: (query: string, signal: AbortSignal, isCurrent: () => boolean) => void | Promise<void>;
}

export function useDebouncedSearch(opts: UseDebouncedSearchOptions) {
	let controller: AbortController | null = null;
	let searchSeq = 0;
	let isSearching = $state(false);

	function isCurrent(seq: number) {
		return seq === searchSeq;
	}

	function cancel() {
		controller?.abort();
		searchSeq++;
		isSearching = false;
	}

	const schedule = debounce((query: string) => {
		if (!opts.canRun() || query !== opts.getQuery().trim()) return;

		void start(query);
	}, opts.debounceMs);

	async function start(query: string) {
		cancel();
		const fresh = new AbortController();

		controller = fresh;
		const mySeq = ++searchSeq;

		isSearching = true;
		try {
			await opts.run(query, fresh.signal, () => isCurrent(mySeq));
		} finally {
			if (isCurrent(mySeq)) isSearching = false;
		}
	}

	return {
		cancel,
		get isSearching() {
			return isSearching;
		},
		run(query: string) {
			schedule(query);
		},
		/** Bump the loading flag synchronously (e.g. before the debounce fires). */
		setLoading(value: boolean) {
			isSearching = value;
		}
	};
}

export type UseDebouncedSearchReturn = ReturnType<typeof useDebouncedSearch>;
