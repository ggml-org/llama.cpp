<script lang="ts">
	import { FolderOpen } from '@lucide/svelte';
	import { ToolsService } from '$lib/services/tools.service';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { BuiltInTool, GlobSearchType, KeyboardKey } from '$lib/enums';
	import {
		abbreviateHome,
		buildCaseInsensitiveGlob,
		joinPath,
		lastPathSegment,
		runGlobSearchWithChildren,
		type GlobEntry
	} from '$lib/utils';
	import * as Popover from '$lib/components/ui/popover';
	import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
	import { useDebouncedSearch } from '$lib/hooks/use-debounced-search.svelte';
	import { usePickerNavigation } from '$lib/hooks/use-picker-navigation.svelte';
	import { useScrollActiveRow } from '$lib/hooks/use-scroll-active-row.svelte';
	import ChatFormWorkingDirectoryChip from './ChatFormWorkingDirectoryChip.svelte';
	import ChatFormWorkingDirectoryResultsList from './ChatFormWorkingDirectoryResultsList.svelte';
	import {
		DEFAULT_MOBILE_BREAKPOINT,
		HOME_TILDE,
		MAX_RESULTS_SHOWN,
		NATIVE_LIMIT,
		NATIVE_MAX_DEPTH,
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
		/**
		 * Controlled open state. The host owns it so both the chip click and
		 * the `/cwd` slash command can open the picker through the same path.
		 */
		isOpen: boolean;
		/**
		 * Two-way bound search query. The host keeps it in sync with the text
		 * after `/cwd ` in the chat input, so typing in either surface updates
		 * the other.
		 */
		query: string;
		/**
		 * Anchor at the top edge of the chat form so the popover floats above
		 * the box, matching the mention picker.
		 */
		customAnchor?: HTMLElement | null;
		onChange?: (directory: string | null) => void;
		/**
		 * Lets the host refocus the chat input so typing can resume without
		 * an extra click after the popover closes.
		 */
		onClose?: () => void;
		/** Fired when the chip is clicked so the host can open the picker. */
		onOpen?: () => void;
	}

	let {
		class: className = '',
		disabled = false,
		directory = null,
		isOpen,
		query = $bindable(''),
		customAnchor = null,
		onChange,
		onClose,
		onOpen
	}: Props = $props();

	// File System Access API is opt-in: when available (Chrome / Edge / Opera) the popover
	// exposes a "Browse" button that opens the native folder picker. When unavailable the
	// popover still works via the text input - no alerts, no upload semantics.
	const pickerSupported =
		typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function';

	let searchInputRef: HTMLInputElement | null = $state(null);

	let queryResults = $state<string[]>([]);
	let searchError = $state<string | null>(null);
	let listContainer = $state<HTMLDivElement | null>(null);

	// Highlight + keyboard-nav state (ArrowUp/Down, Escape, Enter). The
	// scroll trigger is bumped only on keyboard nav, so the results list
	// never auto-scrolls on mouse hover or result replacement.
	const nav = usePickerNavigation({
		isOpen: () => isOpen,
		count: () => queryResults.length,
		onClose: closePicker,
		onSelect: (index) => commit(queryResults[index])
	});

	// Absolute home directory on the server, resolved once per session by
	// the tools store. Anchors both the search scope and the chip's `~`
	// abbreviation.
	let homeBase = $derived(toolsStore.serverHome);

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

	// The search query is owned by the host (two-way bound to the text after
	// `/cwd `). Watch it and run the debounced directory search whenever the
	// picker is open, so typing in either the search input or the chat input
	// drives the same results.
	$effect(() => {
		if (!isOpen) return;
		const q = query.trim();
		nav.reset(-1);
		if (q) {
			search.run(q);
		} else {
			search.cancel();
			queryResults = [];
			searchError = null;
			nav.reset(-1);
			searchScope = homeBase ?? HOME_TILDE;
		}
	});

	// Scrolls the highlighted row into view on keyboard nav only (and when
	// a freshly prioritized list lands). Same behavior ChatFormPickerList
	// provides for its own list.
	useScrollActiveRow({
		getTrigger: () => nav.scrollTrigger,
		getContainer: () => listContainer,
		getIndex: () => nav.hoveredIndex,
		getCount: () => queryResults.length,
		dataIndex: 'result'
	});

	// Effective directory the current search runs against (shown in the
	// footer); updated by the search below, including when an exactly-typed
	// directory is "entered".
	let searchScope = $state(HOME_TILDE);

	// Debounced, abortable directory search backed by the shared cache. The
	// query is two-way bound to the text after `/cwd `, so typing in either
	// the search input or the chat input drives the same results. Stale
	// responses are dropped by the hook's isCurrent guard. An exactly-typed
	// directory is "entered": the shared search lists its children too, so
	// path navigation does not require a trailing slash.
	const search = useDebouncedSearch({
		debounceMs: SEARCH_DEBOUNCE_MS,
		canRun: () => isOpen,
		getQuery: () => query.trim(),
		run: async (q, signal, isCurrent) => {
			const trimmed = q.trim();
			if (!trimmed) {
				queryResults = [];
				searchError = null;
				nav.reset(-1);
				searchScope = homeBase ?? HOME_TILDE;
				return;
			}

			try {
				// Generous limit because ranking happens client-side; only
				// the top MAX_RESULTS_SHOWN are shown.
				const res = await runGlobSearchWithChildren(
					trimmed,
					homeBase ?? HOME_TILDE,
					SEARCH_MAX_DEPTH,
					SEARCH_LIMIT,
					signal,
					{ type: GlobSearchType.DIR }
				);
				if (!isCurrent()) return;
				if (res.error) {
					queryResults = [];
					nav.reset(-1);
					searchError = res.error;
					return;
				}

				searchScope = res.exactDir ?? res.args.path;
				queryResults = res.entries.map((e) => e.path).slice(0, MAX_RESULTS_SHOWN);
				if (queryResults.length > 0) {
					nav.reset(0);
					// new results: scroll the list back to the top (first item is hovered)
					nav.bumpScroll();
				} else {
					nav.reset(-1);
				}
				searchError = null;
			} catch (err) {
				if (!isCurrent() || signal.aborted) return;
				queryResults = [];
				nav.reset(-1);
				searchError = err instanceof Error ? err.message : String(err);
			}
		}
	});
	// Single funnel for every local close so the host refocus fires
	// regardless of which commit/dismiss path ended the interaction.
	function closePicker() {
		onClose?.();
	}

	function commit(path: string) {
		onChange?.(path);
		closePicker();
	}

	function setDirectory(value: string) {
		const trimmed = value.trim();
		if (!trimmed) return;
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
		const value = query.trim();
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
			if (nav.hoveredIndex >= 0 && queryResults[nav.hoveredIndex]) {
				commit(queryResults[nav.hoveredIndex]);
			} else if (queryResults.length === 0) {
				handleSubmit();
			}
		} else if (event.key === KeyboardKey.ARROW_DOWN) {
			if (queryResults.length > 0) {
				event.preventDefault();
				nav.move(1);
			}
		} else if (event.key === KeyboardKey.ARROW_UP) {
			if (queryResults.length > 0) {
				event.preventDefault();
				nav.move(-1);
			}
		}
	}

	function clearDirectory(event?: MouseEvent) {
		// Stop the click from bubbling into the chip button and re-opening
		// the picker on top of the now-cleared state.
		event?.stopPropagation();
		event?.preventDefault();
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
		if (open) {
			void toolsStore.resolveServerHome();
		} else {
			search.cancel();
			// bits-ui-initiated close (Escape on the content, outside-click) -
			// the only path that bypasses closePicker().
			onClose?.();
		}
	}

	// Tooltips only on wider viewports - hover surfaces get in the way on
	// touch / narrow layouts. Mirrors the gate used in ActionIcon.
	let innerWidth = $state(0);
	const showTooltip = $derived(innerWidth > DEFAULT_MOBILE_BREAKPOINT);
</script>

<button
	type="button"
	class={[
		'justify-self-start flex min-w-0 w-auto items-center gap-1 mt-1.5 py-1 px-2 backdrop-blur-2xl rounded-md',
		className
	]}
	onclick={onOpen}
	{disabled}
>
	<ChatFormWorkingDirectoryChip
		{directory}
		{homeBase}
		{disabled}
		{showTooltip}
		onClear={handleDismiss}
	/>
</button>

<Popover.Root open={isOpen} onOpenChange={handleOpenChange}>
	<Popover.Trigger
		class="pointer-events-none absolute inset-0 opacity-0"
		tabindex={-1}
		aria-hidden="true"
	>
		<span class="sr-only">Open working directory picker</span>
	</Popover.Trigger>

	<Popover.Content
		side="top"
		align="start"
		sideOffset={12}
		{customAnchor}
		preventScroll={false}
		onkeydown={handleKeydown}
		onOpenAutoFocus={(event) => event.preventDefault()}
		onCloseAutoFocus={(event) => event.preventDefault()}
		class="w-[var(--bits-popover-anchor-width)] max-w-none rounded-xl border-border/50 p-0 shadow-xl"
	>
		<div class="p-2 min-h-22 flex flex-col justify-between">
			<SearchInput
				bind:ref={searchInputRef}
				bind:value={query}
				placeholder="Choose working directory"
				onClose={closePicker}
				class="w-full"
			/>

			{#if query.trim() && (search.isSearching || queryResults.length > 0 || searchError)}
				<ChatFormWorkingDirectoryResultsList
					results={queryResults}
					hoveredIndex={nav.hoveredIndex}
					isSearching={search.isSearching}
					error={searchError}
					rawQuery={query}
					bind:container={listContainer}
					onCommit={commit}
					onHover={(index) => nav.setHover(index)}
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
				<div class="-mx-2 my-2 h-px bg-border/20" aria-hidden="true"></div>

				<span class="px-2 py-1.5 font-mono text-[10px]">
					Searching in:

					<span class="truncate text-muted-foreground/70" title={searchScope}
						>{abbreviateHome(searchScope, homeBase)}</span
					>
				</span>
			{/if}
		</div>
	</Popover.Content>
</Popover.Root>

<svelte:window bind:innerWidth />
