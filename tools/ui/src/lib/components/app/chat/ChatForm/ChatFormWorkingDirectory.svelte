<script lang="ts">
	import { FolderOpen } from '@lucide/svelte';
	import { untrack } from 'svelte';
	import { SvelteMap } from 'svelte/reactivity';
	import { ToolsService } from '$lib/services/tools.service';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { BuiltInTool, GlobSearchType, KeyboardKey } from '$lib/enums';
	import {
		abbreviateHome,
		buildCaseInsensitiveGlob,
		joinPath,
		lastPathSegment,
		rankEntries,
		splitPathQuery,
		type GlobEntry
	} from '$lib/utils';
	import { debounce } from '$lib/utils/debounce';
	import * as Popover from '$lib/components/ui/popover';
	import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
	import ChatFormWorkingDirectoryChip from './ChatFormWorkingDirectoryChip.svelte';
	import ChatFormWorkingDirectoryResultsList from './ChatFormWorkingDirectoryResultsList.svelte';
	import {
		DEFAULT_MOBILE_BREAKPOINT,
		GLOB_WILDCARD,
		HOME_TILDE,
		MAX_RESULTS_SHOWN,
		NATIVE_LIMIT,
		NATIVE_MAX_DEPTH,
		PATH_NAV_MAX_DEPTH,
		SEARCH_DEBOUNCE_MS,
		SEARCH_LIMIT,
		SEARCH_MAX_DEPTH
	} from '$lib/constants';

	// Microtask delay so the popover's focus scope tears down first.
	const FOCUS_DELAY_MS = 0;

	interface Props {
		class?: string;
		disabled?: boolean;
		directory?: string | null;
		onChange?: (directory: string | null) => void;
		/**
		 * Lets the host refocus the chat input so typing can resume without
		 * an extra click after the popover closes.
		 */
		onClose?: () => void;
	}

	let {
		class: className = '',
		disabled = false,
		directory = $bindable(null),
		onChange,
		onClose
	}: Props = $props();

	// File System Access API is opt-in: when available (Chrome / Edge / Opera) the popover
	// exposes a "Browse" button that opens the native folder picker. When unavailable the
	// popover still works via the text input - no alerts, no upload semantics.
	const pickerSupported =
		typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function';

	// Popover open state; the element handles outside-click and Escape.
	let isOpen = $state(false);
	let inputValue = $state('');
	let searchInputRef: HTMLInputElement | null = $state(null);

	let queryResults = $state<string[]>([]);
	let isSearching = $state(false);
	let searchError = $state<string | null>(null);
	let hoveredIndex = $state(-1);
	// Bumped only by ArrowUp/ArrowDown handlers; the list scrolls the
	// highlighted row into view only via this trigger, never on hover.
	let scrollTrigger = $state(0);
	let listContainer = $state<HTMLDivElement | null>(null);

	// Absolute home directory on the server, resolved once per session by
	// the tools store. Anchors both the search scope and the chip's `~`
	// abbreviation.
	let homeBase = $derived(toolsStore.serverHome);

	// AbortController + sequence counter to discard stale responses when the user
	// keeps typing; a newer call aborts the previous one. The sequence counter
	// also covers the gap between abort and the catch handler.
	let searchController: AbortController | null = null;
	let searchSeq = 0;

	// Cache of the last file_glob_search result per (parent, include, max_depth),
	// so repeated queries in the same directory don't re-walk the tree. Entries
	// expire after a short TTL.
	const SEARCH_CACHE_TTL_MS = 2000;
	const searchCache = new SvelteMap<string, { results: GlobEntry[]; base: string; at: number }>();

	const runSearch = debounce((query: string) => {
		void doSearch(query);
	}, SEARCH_DEBOUNCE_MS);

	// Resolve home eagerly on mount so the chip can abbreviate before the
	// user opens the picker. resolveServerHome() is cached, so repeat calls
	// (e.g. from handleOpenChange) are no-ops.
	$effect(() => {
		if (typeof window === 'undefined') return;
		void toolsStore.resolveServerHome();
	});

	// Auto-focus the search input when the popover opens.
	// HTML `autofocus` is unreliable on dynamically shown elements, so we
	// use a microtask (0ms setTimeout) after the effect flushes.
	$effect(() => {
		if (!isOpen) return;
		setTimeout(() => searchInputRef?.focus(), FOCUS_DELAY_MS);
	});

	let lastScrollTrigger: number | null = null;

	// hoveredIndex/queryResults are untracked so hover and result replacement
	// never re-fire the scroll; keyboard nav is the only path that bumps the trigger
	$effect(() => {
		if (scrollTrigger === lastScrollTrigger) return;
		lastScrollTrigger = scrollTrigger;
		untrack(() => {
			if (!listContainer) return;
			if (hoveredIndex < 0 || hoveredIndex >= queryResults.length) return;
			const selectedElement = listContainer.querySelector(
				`[data-result-index="${hoveredIndex}"]`
			) as HTMLElement | null;
			selectedElement?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
		});
	});

	function cancelSearch() {
		searchController?.abort();
		searchSeq++;
		isSearching = false;
	}

	// Effective directory the current search runs against (shown in the
	// footer); updated by doSearch, including when an exactly-typed
	// directory is "entered".
	let searchScope = $state(HOME_TILDE);

	// Runs a directory listing through the cache, so a repeated query in the
	// same directory does not re-walk the tree on the server.
	async function searchDirs(
		path: string,
		include: string,
		maxDepth: number,
		signal: AbortSignal
	): Promise<{ base: string; entries: GlobEntry[]; error?: string }> {
		const key = `${path}\u0000${include}\u0000${maxDepth}`;
		const cached = searchCache.get(key);
		if (cached && Date.now() - cached.at < SEARCH_CACHE_TTL_MS) {
			return { base: cached.base, entries: cached.results };
		}
		const res = await ToolsService.executeToolRaw(
			BuiltInTool.FILE_GLOB_SEARCH,
			{ path, type: GlobSearchType.DIR, include, max_depth: maxDepth, limit: SEARCH_LIMIT },
			signal
		);
		if (typeof res.error === 'string') return { base: '', entries: [], error: res.error };
		const base = typeof res.base === 'string' ? res.base : '';
		const entries = Array.isArray(res.entries) ? (res.entries as GlobEntry[]) : [];
		searchCache.set(key, { results: entries, base, at: Date.now() });
		return { base, entries };
	}

	async function doSearch(query: string) {
		const trimmed = query.trim();
		if (!trimmed) {
			queryResults = [];
			searchError = null;
			isSearching = false;
			hoveredIndex = -1;
			searchScope = homeBase ?? HOME_TILDE;
			return;
		}

		cancelSearch();
		const controller = new AbortController();
		searchController = controller;
		const mySeq = ++searchSeq;

		const pathQuery = splitPathQuery(trimmed);

		isSearching = true;
		try {
			// A generous limit is requested because ranking happens
			// client-side; only the top 20 are shown.
			const searchPath = pathQuery ? pathQuery.parent : (homeBase ?? HOME_TILDE);
			const include = pathQuery
				? pathQuery.last
					? buildCaseInsensitiveGlob(pathQuery.last)
					: GLOB_WILDCARD
				: buildCaseInsensitiveGlob(trimmed);
			const maxDepth = pathQuery ? PATH_NAV_MAX_DEPTH : SEARCH_MAX_DEPTH;
			const res = await searchDirs(searchPath, include, maxDepth, controller.signal);
			if (mySeq !== searchSeq) return;
			if (res.error) {
				queryResults = [];
				hoveredIndex = -1;
				searchError = res.error;
				return;
			}
			const { base, entries } = res;
			const ranked = rankEntries(entries, pathQuery?.last ?? trimmed);
			let results = ranked.map((e) => joinPath(base, e.path));
			searchScope = pathQuery ? pathQuery.parent : (homeBase ?? HOME_TILDE);

			// An exactly-typed directory is "entered": list its children too,
			// so path navigation doesn't require a trailing slash.
			const last = pathQuery?.last;
			const exact = last
				? ranked.find((e) => lastPathSegment(e.path).toLowerCase() === last.toLowerCase())
				: undefined;
			if (exact) {
				const exactDir = joinPath(base, exact.path);
				const childRes = await searchDirs(
					exactDir,
					GLOB_WILDCARD,
					PATH_NAV_MAX_DEPTH,
					controller.signal
				);
				if (mySeq !== searchSeq) return;
				if (!childRes.error) {
					const children = childRes.entries
						.map((e) => joinPath(childRes.base, e.path))
						.sort((a, b) => a.localeCompare(b));
					results = [...results, ...children];
					searchScope = exactDir;
				}
			}

			queryResults = results.slice(0, MAX_RESULTS_SHOWN);
			hoveredIndex = queryResults.length > 0 ? 0 : -1;
			// new results: scroll the list back to the top (first item is hovered)
			if (hoveredIndex === 0) scrollTrigger++;
			searchError = null;
		} catch (err) {
			if (mySeq !== searchSeq) return;
			queryResults = [];
			hoveredIndex = -1;
			if (controller.signal.aborted) return;
			searchError = err instanceof Error ? err.message : String(err);
		} finally {
			if (mySeq === searchSeq) isSearching = false;
		}
	}

	// Single funnel for every local close so the host refocus fires
	// regardless of which commit/dismiss path ended the interaction.
	function closePicker() {
		isOpen = false;
		onClose?.();
	}

	function commit(path: string) {
		directory = path;
		onChange?.(path);
		closePicker();
	}

	function setDirectory(value: string) {
		const trimmed = value.trim();
		if (!trimmed) return;
		directory = trimmed;
		onChange?.(trimmed);
	}

	// Resolve a folder name picked via the browser-native picker (which exposes
	// only the leaf name) to a server-side absolute path. Returns null when the
	// server cannot locate a matching directory, so the caller can fail visibly
	// instead of committing a bare leaf name that would resolve against the
	// server process working directory.
	async function resolveNativeName(name: string): Promise<string | null> {
		try {
			const res = await ToolsService.executeToolRaw(BuiltInTool.FILE_GLOB_SEARCH, {
				path: homeBase ?? HOME_TILDE,
				type: GlobSearchType.DIR,
				include: buildCaseInsensitiveGlob(name),
				max_depth: NATIVE_MAX_DEPTH,
				limit: NATIVE_LIMIT
			});
			const base = typeof res.base === 'string' ? res.base : '';
			const entries = Array.isArray(res.entries) ? (res.entries as GlobEntry[]) : [];
			const match = entries.find(
				(e) => lastPathSegment(e.path).toLowerCase() === name.toLowerCase()
			);
			return match ? joinPath(base, match.path) : null;
		} catch {
			return null;
		}
	}

	async function browseNative() {
		if (disabled || !window.showDirectoryPicker) return;
		try {
			const handle = await window.showDirectoryPicker();
			const path = await resolveNativeName(handle.name);
			if (path) {
				setDirectory(path);
				closePicker();
			} else {
				// keep the previous cwd and fail visibly instead of committing a
				// bare leaf name that would resolve against the server cwd
				searchError = `Could not resolve "${handle.name}" to a server path`;
			}
		} catch (err) {
			// user cancelled - silently ignore; other errors are logged
			if (err instanceof DOMException && err.name === 'AbortError') return;
			console.error('[ChatFormWorkingDirectory] showDirectoryPicker failed:', err);
		}
	}

	function handleSubmit() {
		const value = inputValue.trim();
		if (!value) {
			closePicker();
			return;
		}
		setDirectory(value);
		closePicker();
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ENTER) {
			event.preventDefault();
			// Commit the highlighted result, falling back to the raw input
			// only when the query returned no matches.
			if (hoveredIndex >= 0 && queryResults[hoveredIndex]) {
				commit(queryResults[hoveredIndex]);
			} else if (queryResults.length === 0) {
				handleSubmit();
			}
		} else if (event.key === KeyboardKey.ARROW_DOWN) {
			if (queryResults.length > 0) {
				event.preventDefault();
				hoveredIndex = (hoveredIndex + 1) % queryResults.length;
				scrollTrigger++;
			}
		} else if (event.key === KeyboardKey.ARROW_UP) {
			if (queryResults.length > 0) {
				event.preventDefault();
				hoveredIndex = hoveredIndex <= 0 ? queryResults.length - 1 : hoveredIndex - 1;
				scrollTrigger++;
			}
		}
	}

	function handleInputInput(value: string) {
		hoveredIndex = -1;
		if (value.trim().length > 0) {
			runSearch(value);
		}
	}

	function clearDirectory(event?: MouseEvent) {
		// Stop the click from bubbling into the popover trigger and re-opening
		// the picker on top of the now-cleared state.
		event?.stopPropagation();
		event?.preventDefault();
		directory = null;
		onChange?.(null);
		closePicker();
	}

	// The chip is always visible; the X clears the directory (no-op when
	// already empty).
	function handleDismiss(event?: MouseEvent) {
		event?.stopPropagation();
		event?.preventDefault();
		if (directory) {
			clearDirectory(event);
		}
	}

	function handleOpenChange(open: boolean) {
		isOpen = open;
		if (open) {
			// Seed the search field with the current path so the user can refine it
			// (or hit Enter to confirm / clear via the X icon).
			inputValue = directory ?? '';
			hoveredIndex = -1;
			queryResults = [];
			searchError = null;
			void toolsStore.resolveServerHome();
			searchScope = homeBase ?? HOME_TILDE;
			if (inputValue.trim()) runSearch(inputValue);
		} else {
			cancelSearch();
			// bits-ui-initiated close (Escape on the content, outside-click,
			// trigger toggle) - the only path that bypasses closePicker().
			onClose?.();
		}
	}

	// Tooltips only on wider viewports - hover surfaces get in the way on
	// touch / narrow layouts. Mirrors the gate used in ActionIcon.
	let innerWidth = $state(0);
	const showTooltip = $derived(innerWidth > DEFAULT_MOBILE_BREAKPOINT);
</script>

<div
	class={[
		'justify-self-start flex min-w-0 w-auto items-center gap-1 mt-1.5 py-1 px-2 backdrop-blur-2xl rounded-md',
		className,
		isOpen && 'w-full'
	]}
>
	<Popover.Root bind:open={isOpen} onOpenChange={handleOpenChange}>
		<Popover.Trigger {disabled} class="flex justify-start">
			<ChatFormWorkingDirectoryChip
				{directory}
				{homeBase}
				{disabled}
				{showTooltip}
				onClear={handleDismiss}
			/>
		</Popover.Trigger>

		<Popover.Content
			side="top"
			align="start"
			sideOffset={4}
			class="md:max-w-3xl w-[calc(100vw-1rem)] rounded-xl border-border/50 p-0 shadow-xl md:-translate-2!"
			onkeydown={handleKeydown}
			onOpenAutoFocus={(event) => event.preventDefault()}
		>
			<div class="p-2 min-h-28 flex flex-col justify-between">
				<SearchInput
					bind:ref={searchInputRef}
					bind:value={inputValue}
					placeholder="Choose working directory"
					onInput={handleInputInput}
					onClose={closePicker}
					class="w-full"
				/>

				{#if inputValue.trim() && (isSearching || queryResults.length > 0 || searchError)}
					<ChatFormWorkingDirectoryResultsList
						results={queryResults}
						{hoveredIndex}
						{isSearching}
						error={searchError}
						rawQuery={inputValue}
						bind:container={listContainer}
						onCommit={commit}
						onHover={(index) => (hoveredIndex = index)}
					/>
				{/if}

				{#if pickerSupported}
					<button
						type="button"
						class="-mt-1 flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none hover:bg-accent hover:text-accent-foreground"
						onclick={browseNative}
					>
						<FolderOpen class="size-4 shrink-0 text-muted-foreground" />
						<span>Browse</span>
					</button>
				{/if}

				{#if homeBase}
					<div class="-mx-2 my-1 h-px bg-border/20" aria-hidden="true"></div>

					<span class="px-2 py-2 font-mono text-[10px]">
						Searching in:

						<span class="truncate text-muted-foreground/70" title={searchScope}
							>{abbreviateHome(searchScope, homeBase)}</span
						>
					</span>
				{/if}
			</div>
		</Popover.Content>
	</Popover.Root>
</div>

<svelte:window bind:innerWidth />
