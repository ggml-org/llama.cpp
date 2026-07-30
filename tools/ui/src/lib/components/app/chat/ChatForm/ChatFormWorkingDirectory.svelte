<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
	import { Folder, FolderOpen, GitBranch, X } from '@lucide/svelte';
	import { untrack } from 'svelte';
	import { fly } from 'svelte/transition';
	import { FilesystemService } from '$lib/services';
	import { abbreviateWorkingDir, ApiError } from '$lib/utils';
	import { debounce } from '$lib/utils/debounce';
	import * as Popover from '$lib/components/ui/popover';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
	import { ActionIcon } from '$lib/components/app/actions';
	import { cn } from '$lib/components/ui/utils';
	import type { ApiFilesystemRoot, ApiFilesystemSearchEntry } from '$lib/types';

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

	// Popover open state. The popover element handles outside-click and Escape;
	// we just react to open and seed the search field with the active path.
	let isOpen = $state(false);
	let inputValue = $state('');
	let searchInputRef: HTMLInputElement | null = $state(null);

	// Search / autocomplete state
	let queryResults = $state<ApiFilesystemSearchEntry[]>([]);
	let isSearching = $state(false);
	let searchError = $state<string | null>(null);
	let endpointDisabled = $state(false);
	let hoveredIndex = $state(-1);
	// Bumped only by ArrowUp/ArrowDown handlers; the list scrolls the
	// highlighted row into view only via this trigger, never on hover.
	let scrollTrigger = $state(0);
	let listContainer = $state<HTMLDivElement | null>(null);
	// Browse roots loaded once per session; default root anchors the search.
	let roots = $state<ApiFilesystemRoot[] | null>(null);
	let loadingRoots = $state(false);
	let rootsError = $state<string | null>(null);

	let defaultRootPath = $derived.by(() => {
		if (!roots || roots.length === 0) return null;
		const def = roots.find((r) => r.default);
		return def ? def.path : roots[0].path;
	});

	// Label on the trigger button: abbreviated active path, or the ghost
	// prompt. The default browse root is intentionally NOT previewed on
	// the chip - the user picks explicitly via the popover.
	let displayLabel = $derived.by(() => {
		if (!directory) return 'Select working directory';
		return abbreviateWorkingDir(directory, roots);
	});

	// Full path surface for the chip - lets the user hover the abbreviated
	// label to recall exactly which directory is set.
	let displayLabelTitle = $derived(directory ?? '');

	// Git metadata for the picked directory. Probed by the server walking up
	// from `directory` looking for `.git/`. Updated whenever the active
	// `directory` changes; stale responses from earlier paths are dropped.
	let gitInfo = $state<{ is_repo: boolean; branch: string } | null>(null);
	let gitController: AbortController | null = null;
	let gitSeq = 0;

	$effect(() => {
		const path = directory;

		// Cancel any in-flight pose for a previous directory before kicking
		// off the next one.
		gitController?.abort();
		gitSeq++;

		if (!path) {
			gitInfo = null;
			return;
		}

		const controller = new AbortController();
		gitController = controller;
		const mySeq = gitSeq;

		FilesystemService.getGitInfo({ path }, controller.signal)
			.then((response) => {
				if (mySeq !== gitSeq) return;
				gitInfo = response.is_repo ? { is_repo: true, branch: response.branch } : null;
			})
			.catch((err: unknown) => {
				if (mySeq !== gitSeq) return;
				// 501 from servers without --tools / --agent is a normal
				// operational state; just hide the branch badge silently.
				if (err instanceof ApiError && err.status === 501) {
					gitInfo = null;
					return;
				}
				if (controller.signal.aborted) return;
				gitInfo = null;
			});
	});

	// AbortController + sequence counter to discard stale responses when the user
	// keeps typing; a newer call aborts the previous one. The sequence counter
	// also covers the gap between abort and the catch handler.
	let searchController: AbortController | null = null;
	let searchSeq = 0;

	const runSearch = debounce((query: string) => {
		void doSearch(query);
	}, 180);

	async function ensureRoots() {
		if (roots !== null || loadingRoots) return;
		loadingRoots = true;
		rootsError = null;
		try {
			const res = await FilesystemService.getRoots();
			roots = res.roots;
		} catch (err) {
			if (err instanceof ApiError && err.status === 501) {
				roots = [];
				endpointDisabled = true;
			} else {
				roots = [];
				rootsError = err instanceof Error ? err.message : String(err);
			}
		} finally {
			loadingRoots = false;
		}
	}

	// Load browse roots eagerly on mount so the trigger can advertise the
	// default browse scope before the user opens the picker. ensureRoots()
	// is idempotent, so the call from handleOpenChange stays a no-op.
	$effect(() => {
		if (typeof window === 'undefined') return;
		void ensureRoots();
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

	async function doSearch(query: string) {
		const trimmed = query.trim();
		if (!trimmed) {
			queryResults = [];
			searchError = null;
			isSearching = false;
			hoveredIndex = -1;
			return;
		}

		cancelSearch();
		const controller = new AbortController();
		searchController = controller;
		const mySeq = ++searchSeq;

		isSearching = true;
		try {
			const response = await FilesystemService.search(
				{
					query: trimmed,
					type: 'directory',
					path: defaultRootPath ?? '',
					limit: 20,
					max_depth: 6,
					show_hidden: true
				},
				controller.signal
			);
			if (mySeq !== searchSeq) return;
			queryResults = response.results;
			hoveredIndex = response.results.length > 0 ? 0 : -1;
			searchError = null;
		} catch (err) {
			if (mySeq !== searchSeq) return;
			queryResults = [];
			hoveredIndex = -1;
			if (controller.signal.aborted) return;
			if (err instanceof ApiError && err.status === 501) {
				endpointDisabled = true;
				searchError = null;
			} else {
				searchError = err instanceof Error ? err.message : String(err);
			}
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

	function commit(entry: ApiFilesystemSearchEntry) {
		directory = entry.path;
		onChange?.(entry.path);
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
		if (endpointDisabled || !defaultRootPath) return name;
		try {
			const res = await FilesystemService.search(
				{
					query: name,
					type: 'directory',
					path: defaultRootPath,
					limit: 1,
					max_depth: 4
				},
				new AbortController().signal
			);
			const match = res.results[0];
			return match && match.name === name ? match.path : name;
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
			void ensureRoots();
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

	// Imperative API: opens the picker without requiring the chip's own
	// trigger to be clicked. Used by ChatForm so picking the "Working
	// Directory" item from the Add dropdown reveals the chip and instantly
	// drops the user into the picker.
	export function openPicker() {
		isOpen = true;
	}

	// Tooltips only on wider viewports - hover surfaces get in the way on
	// touch / narrow layouts. Mirrors the gate used in ActionIcon.
	let innerWidth = $state(0);
	const showTooltip = $derived(innerWidth > 768);

	// Branch label resolved down to a string so the chip's two branches
	// (with / without Tooltip) don't have to re-narrow `gitInfo` inside
	// a snippet body - svelte-check loses the outer narrowing once the
	// markup crosses a Tooltip.Trigger boundary.
	const gitBranchLabel = $derived(gitInfo && gitInfo.is_repo ? gitInfo.branch : '');
</script>

{#snippet resultsList()}
	<div
		bind:this={listContainer}
		class="max-h-48 overflow-y-auto"
		transition:fly={{ y: -4, duration: 100 }}
	>
		{#if isSearching && queryResults.length === 0}
			<div class="px-2 py-1.5 text-sm text-muted-foreground">Searching...</div>
		{:else if endpointDisabled}
			<div class="px-2 py-1.5 text-sm text-muted-foreground">
				Filesystem browsing is disabled. Start the server with
				<code class="rounded bg-muted px-1 py-0.5 text-[10px]">--tools</code>
				or
				<code class="rounded bg-muted px-1 py-0.5 text-[10px]">--agent</code>
				to enable it.
			</div>
		{:else if searchError}
			<div class="px-2 py-1.5 text-sm text-destructive">{searchError}</div>
		{:else if queryResults.length === 0}
			<div class="px-2 py-1.5 text-sm text-muted-foreground">No matching folders</div>
		{:else}
			{#each queryResults as entry, index (entry.path)}
				<button
					type="button"
					data-result-index={index}
					data-highlighted={index === hoveredIndex ? '' : undefined}
					class={cn(
						'relative flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none data-highlighted:bg-accent data-highlighted:text-accent-foreground'
					)}
					onclick={() => commit(entry)}
					onmouseenter={() => (hoveredIndex = index)}
				>
					<Folder class="size-4 shrink-0 text-muted-foreground" />
					<span class="min-w-0 flex-1 truncate font-mono text-left">
						{#each highlightMatch(entry.path, inputValue.trim()) as seg, segIndex (segIndex)}
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

<div class={['justify-self-start flex min-w-0 w-auto items-center gap-1 mt-1.5 py-1 px-2 backdrop-blur-2xl rounded-md', className, isOpen && 'w-full']}>
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

					{#if gitBranchLabel}
						{#if showTooltip}
							<Tooltip.Root>
								<Tooltip.Trigger>
									{#snippet child({ props })}
										<span
											{...props}
											class="inline-flex shrink-0 items-center gap-1 rounded-full bg-muted/70 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
										>
											<GitBranch class="h-2.5 w-2.5" />
											<span>{gitBranchLabel}</span>
										</span>
									{/snippet}
								</Tooltip.Trigger>
								<Tooltip.Content>
									<p>Git branch on disk</p>
								</Tooltip.Content>
							</Tooltip.Root>
						{:else}
							<span
								class="inline-flex shrink-0 items-center gap-1 rounded-full bg-muted/70 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
							>
								<GitBranch class="h-2.5 w-2.5" />
								<span>{gitBranchLabel}</span>
							</span>
						{/if}
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

				{#if inputValue.trim() && (isSearching || queryResults.length > 0 || searchError || endpointDisabled)}
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

				{#if defaultRootPath || rootsError}
					<div class="-mx-2 my-1 h-px bg-border/20" aria-hidden="true"></div>

					{#if defaultRootPath}
						<span class="px-2 py-1.5 font-mono text-[10px]">
							Searching in:

							{#if showTooltip}
								<Tooltip.Root>
									<Tooltip.Trigger>
										{#snippet child({ props })}
											<span {...props} class="truncate text-muted-foreground/70">
												{abbreviateWorkingDir(defaultRootPath, roots)}
											</span>
										{/snippet}
									</Tooltip.Trigger>
									<Tooltip.Content>
										<p>{defaultRootPath}</p>
									</Tooltip.Content>
								</Tooltip.Root>
							{:else}
								<span class="truncate text-muted-foreground/70">
									{abbreviateWorkingDir(defaultRootPath, roots)}
								</span>
							{/if}
						</span>
					{:else if rootsError}
						<div class="px-2 py-1.5 text-xs text-destructive">
							Cannot load browse roots - {rootsError}
						</div>
					{/if}
				{/if}
			</div>
		</Popover.Content>
	</Popover.Root>
</div>

<svelte:window bind:innerWidth />
