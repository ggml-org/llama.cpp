import { debounce } from '$lib/utils/debounce';

/**
 * Shared debounced async-search machinery for the chat-form pickers.
 *
 * Each picker (mention, working-directory) glob-searches the server as the
 * user types. They all need the same low-level plumbing: an AbortController
 * plus a sequence counter to discard stale responses, a debounce, and a
 * live `isSearching` flag. This hook owns that plumbing; the caller supplies
 * a fetcher that performs the network call and commits its own results.
 *
 * The fetcher receives an `isCurrent` callback that returns false once a
 * newer search has started - the fetcher should return early then so it
 * never writes results from a superseded query.
 */

export interface UseDebouncedSearchOptions {
	/** Debounce delay in ms before the debounced fetch fires. */
	debounceMs: number;
	/** Fire-time guard: a scheduled call that outlives a reset is dropped. */
	canRun: () => boolean;
	/** Live query, used to drop a scheduled call whose query changed. */
	getQuery: () => string;
	/**
	 * Perform the search and commit results. Return early when `isCurrent()`
	 * is false, so stale work never lands.
	 */
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
		/** True while a fetch is in flight for the latest query. */
		get isSearching() {
			return isSearching;
		},
		/** Bump the loading flag synchronously (e.g. before the debounce fires). */
		setLoading(value: boolean) {
			isSearching = value;
		},
		/** Schedule a debounced search for `query`. */
		run(query: string) {
			schedule(query);
		},
		/** Abort the in-flight fetch and drop any scheduled call. */
		cancel
	};
}

export type UseDebouncedSearchReturn = ReturnType<typeof useDebouncedSearch>;
