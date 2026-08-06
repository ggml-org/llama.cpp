<script lang="ts">
	import { File, Folder } from '@lucide/svelte';
	import { abbreviateHome, runGlobSearchWithChildren, type GlobEntryResult } from '$lib/utils';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { FileMentionEntryType, GlobSearchType } from '$lib/enums';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import { config } from '$lib/stores/settings.svelte';
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
	 * Floating file/folder mention picker. Opens when the user types
	 * `@<query>` at a token boundary; returns the picked `FileMentionEntry`
	 * via `onSelect` so the parent can splice a `[name](file:///<abs>)<space>`
	 * link into the textarea at the cursor.
	 *
	 * The chat textarea is the search surface: `query` (what the user typed
	 * after `@`) drives a `file_glob_search` tool call scoped to `scopePath`
	 * (the conversation cwd, or the server home when unset). Closes via
	 * Escape, outside-click, or selection. The parent owns the "dismissed
	 * token, don't re-open until it changes" snapshot.
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
		 * Fired when `isOpen` becomes true. The chat textarea is the picker's
		 * "search input", so the host focuses it here to keep the chain
		 * `typed @ -> picker open -> still typing` continuous even if focus
		 * drifted (e.g. closed via outside-click on the chip trigger).
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

	// Coerce the depth setting to a positive integer; an empty/non-numeric
	// value would otherwise reach the server as max_depth 0 = unlimited.
	const searchDepth = $derived.by(() => {
		const n = Number(config().mentionSearchMaxDepth);
		return Number.isInteger(n) && n > 0 ? n : FILE_GLOB_SEARCH_PICKERS_DEFAULT_SEARCH_DEPTH;
	});

	// Absolute home on the server, resolved once per session by the tools
	// store. Anchors the search scope fallback and the `~` abbreviation.
	const home = $derived(toolsStore.serverHome);

	// A smaller window than the WD picker suffices: entries are ranked client-side.
	const MENTION_SEARCH_LIMIT = 50;

	// Debounced, abortable glob search; the fetcher maps each hit to a
	// FileMentionEntry and the hook's isCurrent guard drops stale responses.
	const search = useDebouncedSearch({
		debounceMs: SEARCH_DEBOUNCE_MS,
		canRun: () => isOpen,
		getQuery: () => trimmedQuery,
		run: async (query, signal, isCurrent) => {
			try {
				// A trailing path separator targets a directory, so also list its
				// children. Accept both `/` and `\`.
				const res = await runGlobSearchWithChildren(
					query,
					scopePath ?? home ?? HOME_TILDE,
					searchDepth,
					MENTION_SEARCH_LIMIT,
					signal,
					{ type: GlobSearchType.ALL, descendOnTrailingSeparator: true }
				);
				if (!isCurrent()) return;
				if (res.error) {
					searchResults = [];
					searchError = res.error;
					return;
				}
				const toEntry = (e: GlobEntryResult): FileMentionEntry => ({
					path: e.path,
					name: e.name,
					type: e.type === 'dir' ? FileMentionEntryType.DIRECTORY : FileMentionEntryType.FILE
				});
				searchResults = res.entries.map(toEntry);
				searchError = null;
			} catch (err) {
				if (!isCurrent() || signal.aborted) return;
				searchResults = [];
				searchError = err instanceof Error ? err.message : String(err);
			}
		}
	});

	// A bare `@` with no query is a no-op (the host does not even open the picker).
	const trimmedQuery = $derived((query ?? '').trim());
	const displayedItems = $derived(searchResults);

	const emptyMessage = $derived(
		searchError ? `Search failed - ${searchError}` : 'No matching files or folders'
	);

	// Tooltips only on wider viewports; same gate used elsewhere (ActionIcon, WD chip).
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

	// Keep focus on the chat textarea so typing after `@` flows naturally.
	$effect(() => {
		if (isOpen) onOpened?.();
	});

	// `query` (what the user typed after `@`) drives the debounced fetch.
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
	     We open programmatically via `open={isOpen}`, so it is inert
	     (tabindex=-1 + pointer-events-none + opacity-0 + aria-hidden).
	     Positioning comes from `customAnchor` at the form's top edge. -->
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
		<ChatFormPickerList
			items={displayedItems}
			isLoading={search.isSearching}
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
					{@const Icon = entry.type === FileMentionEntryType.DIRECTORY ? Folder : File}
					<Icon
						class={[
							'mt-0.5 h-4 w-4 shrink-0',
							entry.type === FileMentionEntryType.DIRECTORY
								? 'text-amber-500'
								: 'text-muted-foreground'
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
