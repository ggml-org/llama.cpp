<script lang="ts">
	import { File, Folder } from '@lucide/svelte';
	import { ChatFormPickerList, ChatFormPickerListItem } from '$lib/components/app/chat';
	import HighlightedMatch from '$lib/components/app/forms/HighlightedMatch.svelte';
	import * as Popover from '$lib/components/ui/popover';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import {
		FILE_GLOB_SEARCH_PICKERS_DEFAULT_SEARCH_DEPTH,
		HOME_TILDE,
		SEARCH_DEBOUNCE_MS
	} from '$lib/constants';
	import { BuiltInTool, FileMentionEntryType, GlobSearchType, KeyboardKey } from '$lib/enums';
	import { useDebouncedSearch } from '$lib/hooks/use-debounced-search.svelte';
	import { usePickerNavigation } from '$lib/hooks/use-picker-navigation.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import type { FileMentionEntry } from '$lib/types';
	import { abbreviateHome, type GlobEntryResult, runGlobSearchWithChildren } from '$lib/utils';

	/**
	 * Floating file/folder mention picker. The chat input is the search
	 * surface: `query` (typed after `@`) drives a `file_glob_search` tool
	 * call scoped to `scopePath`. The parent owns the "dismissed token,
	 * don't re-open until it changes" snapshot.
	 */
	interface Props {
		class?: string;
		isOpen: boolean;
		query: string;
		customAnchor?: HTMLElement | null;
		scopePath?: string | null;
		onClose: () => void;
		onSelect: (entry: FileMentionEntry) => void;
		/** Fired when `isOpen` becomes true, so the host can keep focus on the chat input. */
		onOpened?: () => void;
	}

	let {
		class: className = '',
		customAnchor = null,
		isOpen,
		onClose,
		onOpened,
		onSelect,
		query,
		scopePath = null
	}: Props = $props();

	const nav = usePickerNavigation({
		count: () => displayedItems.length,
		isOpen: () => isOpen,
		onClose: () => onClose(),
		onSelect: (index) => handleSelect(displayedItems[index])
	});

	// When the server does not expose file_glob_search (started without
	// --tools) or the user disabled it, the picker still opens but explains
	// why instead of firing searches that would only fail.
	const fileSearchKey = $derived(toolsStore.getPermissionKey(BuiltInTool.FILE_GLOB_SEARCH));
	const fileSearchEnabled = $derived(
		fileSearchKey !== null && toolsStore.isToolEnabled(fileSearchKey)
	);

	let searchResults = $state<FileMentionEntry[]>([]);
	let searchError = $state<string | null>(null);

	// Coerce the depth setting to a positive integer; an invalid value
	// would otherwise reach the server as max_depth 0 = unlimited.
	const searchDepth = $derived.by(() => {
		const n = Number(config().mentionSearchMaxDepth);

		return Number.isInteger(n) && n > 0 ? n : FILE_GLOB_SEARCH_PICKERS_DEFAULT_SEARCH_DEPTH;
	});

	const home = $derived(toolsStore.serverHome);

	// A smaller window than the WD picker suffices: entries are ranked client-side.
	const MENTION_SEARCH_LIMIT = 50;

	const search = useDebouncedSearch({
		canRun: () => isOpen && fileSearchEnabled,
		debounceMs: SEARCH_DEBOUNCE_MS,
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
					{ descendOnTrailingSeparator: true, type: GlobSearchType.ALL }
				);

				if (!isCurrent()) return;

				if (res.error) {
					searchResults = [];
					searchError = res.error;

					return;
				}

				const toEntry = (e: GlobEntryResult): FileMentionEntry => ({
					name: e.name,
					path: e.path,
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

	const trimmedQuery = $derived((query ?? '').trim());
	const displayedItems = $derived(searchResults);

	const emptyMessage = $derived.by(() => {
		if (fileSearchKey === null) {
			return 'File search is unavailable on this server (started without --tools)';
		}

		if (!fileSearchEnabled) {
			return 'File search is disabled - enable "Search files" in Settings > Tools to use @-mentions';
		}

		return searchError ? `Search failed - ${searchError}` : 'No matching files or folders';
	});

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

	$effect(() => {
		if (isOpen) onOpened?.();
	});

	$effect(() => {
		const q = (query ?? '').trim();

		if (!isOpen || !q || !fileSearchEnabled) {
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
		// Always consume Enter while the picker is open - even with no
		// result yet (skeletons) or no matches - so the chat form's
		// Enter-to-submit never fires mid-search.
		if (isOpen && event.key === KeyboardKey.ENTER) {
			event.preventDefault();

			if (nav.hoveredIndex >= 0 && displayedItems[nav.hoveredIndex]) {
				handleSelect(displayedItems[nav.hoveredIndex]);
			}

			return true;
		}

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
