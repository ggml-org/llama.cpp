<script lang="ts">
	import { Folder, FolderOpen, X } from '@lucide/svelte';
	import { untrack } from 'svelte';
	import { fly } from 'svelte/transition';
	import { ToolsService } from '$lib/services/tools.service';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { BuiltInTool } from '$lib/enums';
	import { abbreviateWorkingDir, abbreviateHome, lastPathSegment } from '$lib/utils';
	import { debounce } from '$lib/utils/debounce';
	import * as Popover from '$lib/components/ui/popover';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
	import { ActionIcon } from '$lib/components/app/actions';
	import { cn } from '$lib/components/ui/utils';

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

	// One entry of file_glob_search's structured `entries` result field.
	interface GlobEntry {
		path: string;
		type: string;
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

	// Popover open state. The popover element handles outside-click and Escape;
	// we just react to open and seed the search field with the active path.
	let isOpen = $state(false);
	let inputValue = $state('');
	let searchInputRef: HTMLInputElement | null = $state(null);

	// Search / autocomplete state. Results are absolute directory paths.
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

	// Label on the trigger button: abbreviated active path, or the ghost
	// prompt.
	let displayLabel = $derived.by(() => {
		if (!directory) return 'Select working directory';
		return abbreviateWorkingDir(directory, homeBase);
	});

	// Full path surface for the chip - lets the user hover the abbreviated
	// label to recall exactly which directory is set.
	let displayLabelTitle = $derived(directory ?? '');

	// AbortController + sequence counter to discard stale responses when the user
	// keeps typing; a newer call aborts the previous one. The sequence counter
	// also covers the gap between abort and the catch handler.
	let searchController: AbortController | null = null;
	let searchSeq = 0;

	const runSearch = debounce((query: string) => {
		void doSearch(query);
	}, 180);

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
		setTimeout(() => searchInputRef?.focus(), 0);
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

	// Build a substring glob from the raw query: letters become
	// case-insensitive char classes (glob_match supports [xX]) so "proj"
	// matches "Project-Alpha"; glob metacharacters are dropped.
	// Path-like queries (starting with / or ~) are treated as path
	// navigation: search the parent directory for the last segment instead
	// of glob-matching the whole query against home-relative entries.
	function splitPathQuery(query: string): { parent: string; last: string } | null {
		if (!query.startsWith('/') && !query.startsWith('~')) return null;
		const normalized = query.replace(/\/+$/, '');
		if (!normalized || normalized === '~') {
			return { parent: normalized === '~' ? '~' : '/', last: '' };
		}
		const idx = normalized.lastIndexOf('/');
		if (idx === 0) return { parent: '/', last: normalized.slice(1) };
		return { parent: normalized.slice(0, idx), last: normalized.slice(idx + 1) };
	}

	// Effective directory the current search runs against (shown in the
	// footer); updated by doSearch, including when an exactly-typed
	// directory is "entered".
	let searchScope = $state('~');

	function buildCaseInsensitiveGlob(query: string): string {
		let out = '*';
		for (const c of query) {
			const lo = c.toLowerCase();
			const up = c.toUpperCase();
			if (lo !== up) out += `[${lo}${up}]`;
			else if (!'*?[]'.includes(c)) out += c;
		}
		return out + '*';
	}

	// Client-side ranking: exact basename match first, then prefix, then
	// substring; ties broken by shorter path, then alphabetically.
	function rankScore(path: string, query: string): number {
		const name = lastPathSegment(path).toLowerCase();
		const q = query.toLowerCase();
		if (name === q) return 0;
		if (name.startsWith(q)) return 1;
		if (name.includes(q)) return 2;
		return 3;
	}

	function rankEntries(entries: GlobEntry[], query: string): GlobEntry[] {
		return [...entries].sort(
			(a, b) =>
				rankScore(a.path, query) - rankScore(b.path, query) ||
				a.path.length - b.path.length ||
				a.path.localeCompare(b.path)
		);
	}

	function joinPath(base: string, rel: string): string {
		if (!base) return rel;
		return base.replace(/\/+$/, '') + '/' + rel;
	}

	async function doSearch(query: string) {
		const trimmed = query.trim();
		if (!trimmed) {
			queryResults = [];
			searchError = null;
			isSearching = false;
			hoveredIndex = -1;
			searchScope = homeBase ?? '~';
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
			const res = await ToolsService.executeToolRaw(
				BuiltInTool.FILE_GLOB_SEARCH,
				{
					path: pathQuery ? pathQuery.parent : (homeBase ?? '~'),
					type: 'dir',
					include: pathQuery
						? pathQuery.last
							? buildCaseInsensitiveGlob(pathQuery.last)
							: '*'
						: buildCaseInsensitiveGlob(trimmed),
					max_depth: pathQuery ? 1 : 6,
					limit: 100
				},
				controller.signal
			);
			if (mySeq !== searchSeq) return;
			if (typeof res.error === 'string') {
				queryResults = [];
				hoveredIndex = -1;
				searchError = res.error;
				return;
			}
			const base = typeof res.base === 'string' ? res.base : '';
			const entries = Array.isArray(res.entries) ? (res.entries as GlobEntry[]) : [];
			const ranked = rankEntries(entries, pathQuery?.last ?? trimmed);
			let results = ranked.map((e) => joinPath(base, e.path));
			searchScope = pathQuery ? pathQuery.parent : (homeBase ?? '~');

			// An exactly-typed directory is "entered": list its children too,
			// so path navigation doesn't require a trailing slash.
			const last = pathQuery?.last;
			const exact = last
				? ranked.find((e) => lastPathSegment(e.path).toLowerCase() === last.toLowerCase())
				: undefined;
			if (exact) {
				const exactDir = joinPath(base, exact.path);
				const childRes = await ToolsService.executeToolRaw(
					BuiltInTool.FILE_GLOB_SEARCH,
					{ path: exactDir, type: 'dir', include: '*', max_depth: 1, limit: 100 },
					controller.signal
				);
				if (mySeq !== searchSeq) return;
				if (typeof childRes.error !== 'string') {
					const childBase = typeof childRes.base === 'string' ? childRes.base : '';
					const childEntries = Array.isArray(childRes.entries)
						? (childRes.entries as GlobEntry[])
						: [];
					const children = childEntries
						.map((e) => joinPath(childBase, e.path))
						.sort((a, b) => a.localeCompare(b));
					results = [...results, ...children];
					searchScope = exactDir;
				}
			}

			queryResults = results.slice(0, 20);
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
	// only the leaf name) to a server-side absolute path. Falls back to the
	// leaf name when the server cannot locate a matching directory.
	async function resolveNativeName(name: string): Promise<string> {
		try {
			const res = await ToolsService.executeToolRaw(BuiltInTool.FILE_GLOB_SEARCH, {
				path: homeBase ?? '~',
				type: 'dir',
				include: buildCaseInsensitiveGlob(name),
				max_depth: 4,
				limit: 20
			});
			const base = typeof res.base === 'string' ? res.base : '';
			const entries = Array.isArray(res.entries) ? (res.entries as GlobEntry[]) : [];
			const match = entries.find(
				(e) => lastPathSegment(e.path).toLowerCase() === name.toLowerCase()
			);
			return match ? joinPath(base, match.path) : name;
		} catch {
			return name;
		}
	}

	async function browseNative() {
		if (disabled || !window.showDirectoryPicker) return;
		try {
			const handle = await window.showDirectoryPicker();
			const path = await resolveNativeName(handle.name);
			setDirectory(path);
			closePicker();
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
		if (event.key === 'Enter') {
			event.preventDefault();
			// Commit the highlighted search result when there is one
			// (the user may have arrow-keyed to it or it is the auto-
			// selected first row after results landed). Fall back to
			// the raw input only when the query returned no matches,
			// so the user can still type a known absolute path.
			if (hoveredIndex >= 0 && queryResults[hoveredIndex]) {
				commit(queryResults[hoveredIndex]);
			} else if (queryResults.length === 0) {
				handleSubmit();
			}
		} else if (event.key === 'ArrowDown') {
			if (queryResults.length > 0) {
				event.preventDefault();
				hoveredIndex = (hoveredIndex + 1) % queryResults.length;
				scrollTrigger++;
			}
		} else if (event.key === 'ArrowUp') {
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

	// Chip is always visible - the X just clears the picked directory and
	// reveals the empty "Select working directory" placeholder again. No-op
	// when there's already nothing to clear.
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
			searchScope = homeBase ?? '~';
			// show the current directory (and its siblings) right away
			if (inputValue.trim()) runSearch(inputValue);
		} else {
			cancelSearch();
			// bits-ui-initiated close (Escape on the content, outside-click,
			// trigger toggle) - the only path that bypasses closePicker().
			onClose?.();
		}
	}

	// Splits `text` into alternating segments at each case-insensitive
	// occurrence of `query`. Used by the results list to highlight the search
	// terms inside full-path strings.
	function highlightMatch(text: string, query: string): { text: string; match: boolean }[] {
		if (!query) return [{ text, match: false }];
		const segments: { text: string; match: boolean }[] = [];
		const lowerText = text.toLowerCase();
		const lowerQuery = query.toLowerCase();
		let i = 0;
		while (i < text.length) {
			const idx = lowerText.indexOf(lowerQuery, i);
			if (idx < 0) {
				segments.push({ text: text.slice(i), match: false });
				break;
			}
			if (idx > i) segments.push({ text: text.slice(i, idx), match: false });
			segments.push({ text: text.slice(idx, idx + query.length), match: true });
			i = idx + query.length;
		}
		return segments;
	}

	// Tooltips only on wider viewports - hover surfaces get in the way on
	// touch / narrow layouts. Mirrors the gate used in ActionIcon.
	let innerWidth = $state(0);
	const showTooltip = $derived(innerWidth > 768);
</script>

{#snippet resultsList()}
	<div
		bind:this={listContainer}
		class="max-h-48 overflow-y-auto"
		transition:fly={{ y: -4, duration: 100 }}
	>
		{#if isSearching && queryResults.length === 0}
			<div class="px-2 py-1.5 text-sm text-muted-foreground">Searching...</div>
		{:else if searchError}
			<div class="px-2 py-1.5 text-sm text-destructive">{searchError}</div>
		{:else if queryResults.length === 0}
			<div class="px-2 py-1.5 text-sm text-muted-foreground">No matching folders</div>
		{:else}
			{#each queryResults as path, index (path)}
				<button
					type="button"
					data-result-index={index}
					data-highlighted={index === hoveredIndex ? '' : undefined}
					class={cn(
						'relative flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none data-highlighted:bg-accent data-highlighted:text-accent-foreground'
					)}
					onclick={() => commit(path)}
					onmouseenter={() => (hoveredIndex = index)}
				>
					<Folder class="size-4 shrink-0 text-muted-foreground" />
					<span class="min-w-0 flex-1 truncate font-mono text-left">
						{#each highlightMatch(path, inputValue.trim()) as seg, segIndex (segIndex)}
							{#if seg.match}
								<mark class="rounded bg-yellow-200/60 px-0.5 text-foreground dark:bg-yellow-500/30"
									>{seg.text}</mark
								>
							{:else}
								{seg.text}
							{/if}
						{/each}
					</span>
				</button>
			{/each}
		{/if}
	</div>
{/snippet}

<div
	class={[
		'justify-self-start flex min-w-0 w-auto items-center gap-1 mt-1.5 py-1 px-2 backdrop-blur-2xl rounded-md',
		className,
		isOpen && 'w-full'
	]}
>
	<Popover.Root bind:open={isOpen} onOpenChange={handleOpenChange}>
		<Popover.Trigger {disabled} class="flex justify-start">
			<span
				class="text-muted-foreground inline-flex items-center gap-1 text-xs group"
				class:text-foreground={directory}
			>
				<div class="flex min-w-0 items-center gap-1 cursor-pointer">
					<Folder class="w-3.5 h-3.5" />

					{#if showTooltip && displayLabelTitle}
						<Tooltip.Root>
							<Tooltip.Trigger>
								{#snippet child({ props })}
									<span {...props} class="max-w-64 truncate">{displayLabel}</span>
								{/snippet}
							</Tooltip.Trigger>
							<Tooltip.Content>
								<p>{displayLabelTitle}</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{:else}
						<span class="max-w-64 truncate">{displayLabel}</span>
					{/if}
				</div>

				{#if directory}
					<div
						class="w-0 overflow-hidden opacity-0 transition-[width,opacity] duration-200 ease-out group-hover:w-auto group-hover:opacity-100"
					>
						<ActionIcon
							icon={X}
							tooltip="Reset working directory"
							ariaLabel="Reset working directory"
							{disabled}
							onclick={handleDismiss}
							iconSize="h-3 w-3"
							stopPropagationOnClick
							class="!h-4 !w-4 shrink-0 text-muted-foreground hover:text-foreground"
						/>
					</div>
				{/if}
			</span>
		</Popover.Trigger>

		<Popover.Content
			side="top"
			align="start"
			sideOffset={4}
			class="w-[var(--bits-popover-anchor-width)] min-w-md max-w-none rounded-xl border-border/50 p-0 shadow-xl"
			onkeydown={handleKeydown}
			onOpenAutoFocus={(event) => event.preventDefault()}
		>
			<div class="p-2">
				<SearchInput
					bind:ref={searchInputRef}
					bind:value={inputValue}
					placeholder="Choose working directory"
					onInput={handleInputInput}
					onClose={closePicker}
					class="w-full"
				/>

				{#if inputValue.trim() && (isSearching || queryResults.length > 0 || searchError)}
					{@render resultsList()}
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
</div>

<svelte:window bind:innerWidth />
