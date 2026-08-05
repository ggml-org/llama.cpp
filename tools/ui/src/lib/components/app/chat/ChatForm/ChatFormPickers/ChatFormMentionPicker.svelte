<script lang="ts">
	import { File, Folder } from '@lucide/svelte';
	import {
		abbreviateHome,
		buildGlobSearchArgs,
		joinPath,
		lastPathSegment,
		rankEntries,
		runGlobSearch
	} from '$lib/utils';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { GlobSearchType } from '$lib/enums';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { recentMentionsStore } from '$lib/stores/recent-mentions.svelte';
	import * as Popover from '$lib/components/ui/popover';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import HighlightedMatch from '$lib/components/app/forms/HighlightedMatch.svelte';
	import { ChatFormPickerList, ChatFormPickerListItem } from '$lib/components/app/chat';
	import { useDebouncedSearch } from '$lib/hooks/use-debounced-search.svelte';
	import { usePickerNavigation } from '$lib/hooks/use-picker-navigation.svelte';
	import type { FileMentionEntry } from '$lib/types';
	import {
		FILE_GLOB_SEARCH_PICKERS_DEFAULT_SEARCH_DEPTH,
		HOME_TILDE,
		SEARCH_DEBOUNCE_MS
	} from '$lib/constants';

	/**
	 * Floating file/folder mention picker.
	 *
	 * Opens when the user types `@<query>` at a token boundary inside the
	 * chat textarea. Returns the picked `FileMentionEntry` via `onSelect`
	 * so the parent can splice a `[name](file:///<abs>)<space>` markdown
	 * link into the textarea at the cursor.
	 *
	 * The picker has no internal search input - the chatbot textarea is
	 * the search surface, and `query` (what the user typed after `@`)
	 * drives a `file_glob_search` tool call scoped to `scopePath` (the
	 * conversation cwd, or the server home when unset). Closes via
	 * Escape, outside-click, or selection. The parent owns the "user
	 * dismissed this token, don't re-open until it changes" snapshot so
	 * the picker stays simple.
	 */
	interface Props {
		class?: string;
		isOpen: boolean;
		query: string;
		customAnchor?: HTMLElement | null;
		scopePath?: string | null;
		onClose: () => void;
		onSelect: (entry: FileMentionEntry) => void;
		/**
		 * Fired when `isOpen` becomes true. The chat textarea is the
		 * picker's "search input", so the host focuses it here to keep
		 * the chain `typed @ -> picker open -> still typing` continuous
		 * even if focus drifted (e.g. closed via outside-click on the
		 * chip trigger the next time the picker re-opens).
		 */
		onOpened?: () => void;
	}

	let {
		class: className = '',
		isOpen,
		query,
		customAnchor = null,
		scopePath = null,
		onClose,
		onSelect,
		onOpened
	}: Props = $props();

	const nav = usePickerNavigation({
		isOpen: () => isOpen,
		count: () => displayedItems.length,
		onClose: () => onClose(),
		onSelect: (index) => handleSelect(displayedItems[index])
	});

	let searchResults = $state<FileMentionEntry[]>([]);
	let searchError = $state<string | null>(null);

	// Search depth from settings, coerced to a valid positive integer.
	// The setting can hold an empty string or other non-numeric value
	// (e.g. an empty input field), which the server would interpret as
	// max_depth 0 = unlimited and walk the whole tree. Fall back to
	// FILE_GLOB_SEARCH_PICKERS_DEFAULT_SEARCH_DEPTH for any invalid value.
	const searchDepth = $derived.by(() => {
		const n = Number(config().mentionSearchMaxDepth);
		return Number.isInteger(n) && n > 0 ? n : FILE_GLOB_SEARCH_PICKERS_DEFAULT_SEARCH_DEPTH;
	});

	// Absolute home directory on the server, resolved once per session by
	// the tools store. Anchors the search scope fallback and the `~`
	// abbreviation of result paths.
	const home = $derived(toolsStore.serverHome);

	// The mention search shows a focused list; a smaller window than the
	// working-directory picker is enough because entries are ranked client-side.
	const MENTION_SEARCH_LIMIT = 50;

	// Debounced, abortable glob search. The chat textarea is the search
	// surface, so the fetcher maps each hit to a FileMentionEntry for the
	// picker list; stale responses are dropped by the hook's isCurrent guard.
	const search = useDebouncedSearch({
		debounceMs: SEARCH_DEBOUNCE_MS,
		canRun: () => isOpen,
		getQuery: () => trimmedQuery,
		run: async (query, signal, isCurrent) => {
			const args = buildGlobSearchArgs(query, scopePath ?? home ?? HOME_TILDE, searchDepth);
			try {
				const res = await runGlobSearch(args, GlobSearchType.ALL, MENTION_SEARCH_LIMIT, signal);
				if (!isCurrent()) return;
				if (res.error) {
					searchResults = [];
					searchError = res.error;
					return;
				}
				searchResults = rankEntries(res.entries, args.rankQuery).map((e) => {
					const path = joinPath(res.base, e.path);
					return {
						path,
						name: lastPathSegment(e.path),
						type: e.type === 'dir' ? 'directory' : 'file'
					};
				});
				searchError = null;
			} catch (err) {
				if (!isCurrent() || signal.aborted) return;
				searchResults = [];
				searchError = err instanceof Error ? err.message : String(err);
			}
		}
	});

	// Most-recently-picked entries (deduped, capped, persisted to
	// localStorage). Surfaced when the user opens the picker with no
	// characters typed after `@`, so they can re-use a file or folder
	// without re-typing the search.
	const recentMentions = $derived(recentMentionsStore.items);

	// What the list actually renders. Recents when the user has not
	// typed anything after `@`, live search results otherwise.
	const trimmedQuery = $derived((query ?? '').trim());
	const isShowingRecents = $derived(trimmedQuery === '');
	const displayedItems = $derived(isShowingRecents ? recentMentions : searchResults);

	// Empty-message policy:
	//  - recents empty -> nudge them to start typing
	//  - search error -> surface it (network / scope issues)
	//  - search returned nothing -> "No matching files or folders"
	const emptyMessage = $derived(
		isShowingRecents
			? 'Start typing to search files and folders'
			: searchError
				? `Search failed - ${searchError}`
				: 'No matching files or folders'
	);

	// Tooltips only on wider viewports - hover surfaces get in the way on
	// touch / narrow layouts. Same gate used elsewhere (ActionIcon, WD chip).
	const showTooltip = $derived(!isMobile.current);

	$effect(() => {
		if (typeof window === 'undefined') return;
		void toolsStore.resolveServerHome();
	});

	$effect(() => {
		if (isOpen) {
			nav.reset(0);
		}
	});

	// Fire `onOpened()` whenever the picker transitions to open. Keeps
	// focus on the chat form textarea so typing after `@` flows naturally;
	// reading only `isOpen` makes `onOpened` itself opaque to the
	// reactive system, so hover / query churn cannot cause re-focus
	// storms while the picker is already open.
	$effect(() => {
		if (isOpen) onOpened?.();
	});

	// The chat textarea is the search surface: `query` (what the user typed
	// after `@`) drives the debounced fetch directly. Empty query shows the
	// recents panel instead of searching.
	$effect(() => {
		const q = (query ?? '').trim();
		if (!isOpen || !q) {
			search.cancel();
			searchResults = [];
			searchError = null;
			return;
		}
		search.setLoading(true);
		search.run(q);
	});

	function handleSelect(entry: FileMentionEntry) {
		// Bump to the front of the recent-mentions list before handing
		// off so a re-pick of the same entry within the same mount
		// already sees it at the top of the recents panel.
		recentMentionsStore.add(entry);
		onSelect(entry);
		onClose();
	}

	export function handleKeydown(event: KeyboardEvent): boolean {
		return nav.handleKeydown(event);
	}
</script>

<Popover.Root
	open={isOpen}
	onOpenChange={(open) => {
		if (!open) onClose();
	}}
>
	<!-- Invisible form-wide trigger: stops bits-ui's outside-click detector
	     from closing the picker when the user clicks inside the textarea.
	     We DO NOT use this trigger for opening (we open programmatically via
	     `open={isOpen}`) so it's tabindex=-1 + pointer-events-none + opacity-0
	     + aria-hidden. Positioning comes from `customAnchor` below, which
	     sits at the form's top edge so the popover floats above the box. -->
	<Popover.Trigger
		class="pointer-events-none absolute inset-0 opacity-0"
		tabindex={-1}
		aria-hidden="true"
	>
		<span class="sr-only">Open file mention picker</span>
	</Popover.Trigger>

	<Popover.Content
		align="start"
		side="top"
		sideOffset={12}
		{customAnchor}
		preventScroll={false}
		onkeydown={handleKeydown}
		onOpenAutoFocus={(event) => event.preventDefault()}
		onCloseAutoFocus={(event) => event.preventDefault()}
		class={[
			'w-[var(--bits-popover-anchor-width)] max-w-none rounded-xl border-border/50 p-0 shadow-xl',
			className
		]}
	>
		{#if isShowingRecents && recentMentions.length > 0}
			<div
				class="flex items-center justify-between px-4 pt-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
			>
				<span>Recently used</span>
			</div>
		{/if}

		<ChatFormPickerList
			items={displayedItems}
			isLoading={isShowingRecents ? false : search.isSearching}
			selectedIndex={nav.hoveredIndex}
			showSearchInput={false}
			searchQuery={query ?? ''}
			{emptyMessage}
			itemKey={(entry) => entry.type + ':' + entry.path}
			scrollTrigger={nav.scrollTrigger}
		>
			{#snippet item(entry, index, isSelected)}
				<ChatFormPickerListItem
					dataIndex={index}
					{isSelected}
					onclick={() => handleSelect(entry)}
					onmouseenter={() => nav.setHover(index)}
				>
					{@const Icon = entry.type === 'directory' ? Folder : File}
					<Icon
						class={[
							'mt-0.5 h-4 w-4 shrink-0',
							entry.type === 'directory' ? 'text-amber-500' : 'text-muted-foreground'
						]}
					/>
					<div class="flex min-w-0 flex-1 flex-col">
						<div class="flex min-w-0 items-center gap-2">
							{#if showTooltip}
								<Tooltip.Root>
									<Tooltip.Trigger>
										{#snippet child({ props })}
											<span {...props} class="truncate text-sm font-medium">{entry.name}</span>
										{/snippet}
									</Tooltip.Trigger>
									<Tooltip.Content>
										<p>{entry.path}</p>
									</Tooltip.Content>
								</Tooltip.Root>
							{:else}
								<span class="truncate text-sm font-medium">{entry.name}</span>
							{/if}
							<span
								class="shrink-0 rounded-full bg-muted px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wide text-muted-foreground"
							>
								{entry.type}
							</span>
						</div>
						<span class="min-w-0 flex-1 truncate font-mono text-left text-xs">
							<HighlightedMatch text={abbreviateHome(entry.path, home)} query={trimmedQuery} />
						</span>
					</div>
				</ChatFormPickerListItem>
			{/snippet}
		</ChatFormPickerList>
	</Popover.Content>
</Popover.Root>
