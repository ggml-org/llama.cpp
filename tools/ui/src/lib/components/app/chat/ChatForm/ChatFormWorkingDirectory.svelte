<script lang="ts">
	import ChatFormWorkingDirectoryChip from './ChatFormWorkingDirectoryChip.svelte';
	import ChatFormWorkingDirectoryResultsList from './ChatFormWorkingDirectoryResultsList.svelte';
	import { FolderOpen } from '@lucide/svelte';
	import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
	import * as Popover from '$lib/components/ui/popover';
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
	import { BuiltInTool, GlobSearchType, KeyboardKey } from '$lib/enums';
	import { useDebouncedSearch } from '$lib/hooks/use-debounced-search.svelte';
	import { usePickerNavigation } from '$lib/hooks/use-picker-navigation.svelte';
	import { useScrollActiveRow } from '$lib/hooks/use-scroll-active-row.svelte';
	import { ToolsService } from '$lib/services/tools.service';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import {
		abbreviateHome,
		buildCaseInsensitiveGlob,
		type GlobEntry,
		joinPath,
		lastPathSegment,
		runGlobSearchWithChildren
	} from '$lib/utils';

	// Microtask delay so the popover's focus scope tears down first.
	const FOCUS_DELAY_MS = 0;

	interface Props {
		class?: string;
		disabled?: boolean;
		directory?: string | null;
		/** Controlled open state; the host owns it so the chip click and the
		 * `/cwd` slash command open the picker through the same path. */
		isOpen: boolean;
		/** Two-way bound query, kept in sync with the text after `/cwd `. */
		query: string;
		/** Anchor at the form's top edge so the popover floats above the box. */
		customAnchor?: HTMLElement | null;
		onChange?: (directory: string | null) => void;
		/** Lets the host refocus the chat input after the popover closes. */
		onClose?: () => void;
		/** Fired when the chip is clicked so the host can open the picker. */
		onOpen?: () => void;
	}

	let {
		class: className = '',
		customAnchor = null,
		directory = null,
		disabled = false,
		isOpen,
		onChange,
		onClose,
		onOpen,
		query = $bindable('')
	}: Props = $props();

	// File System Access API is opt-in (Chrome / Edge / Opera): the popover
	// exposes a "Browse" button only when available.
	const pickerSupported =
		typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function';

	// When the server does not serve file_glob_search or the user disabled
	// it, the picker still opens for manual entry but explains why search is
	// unavailable instead of firing searches that would only fail. Browse is
	// hidden too: it resolves the picked folder name through the same tool.
	const fileSearchKey = $derived(toolsStore.getPermissionKey(BuiltInTool.FILE_GLOB_SEARCH));
	const fileSearchEnabled = $derived(
		fileSearchKey !== null && toolsStore.isToolEnabled(fileSearchKey)
	);
	const searchUnavailableMessage = $derived(
		fileSearchKey === null
			? 'File search is unavailable on this server - type a full path and press Enter'
			: 'File search is disabled - type a full path and press Enter, or enable "Search files" in Settings > Tools'
	);

	let searchInputRef: HTMLInputElement | null = $state(null);

	let queryResults = $state<string[]>([]);
	let searchError = $state<string | null>(null);
	let listContainer = $state<HTMLDivElement | null>(null);

	const nav = usePickerNavigation({
		count: () => queryResults.length,
		isOpen: () => isOpen,
		onClose: closePicker,
		onSelect: (index) => commit(queryResults[index])
	});

	let homeBase = $derived(toolsStore.serverHome);

	// Resolve home eagerly so the chip can abbreviate before the picker opens.
	$effect(() => {
		if (typeof window === 'undefined') return;

		void toolsStore.resolveServerHome();
	});

	// HTML `autofocus` is unreliable on dynamically shown elements.
	$effect(() => {
		if (!isOpen) return;

		setTimeout(() => searchInputRef?.focus(), FOCUS_DELAY_MS);
	});

	$effect(() => {
		if (!isOpen) return;

		const q = query.trim();

		nav.reset(-1);

		if (q && fileSearchEnabled) {
			search.run(q);
		} else {
			search.cancel();
			queryResults = [];
			searchError = null;
			nav.reset(-1);
			searchScope = homeBase ?? HOME_TILDE;
		}
	});

	useScrollActiveRow({
		dataIndex: 'result',
		getContainer: () => listContainer,
		getCount: () => queryResults.length,
		getIndex: () => nav.hoveredIndex,
		getTrigger: () => nav.scrollTrigger
	});

	let searchScope = $state(HOME_TILDE);

	// An exactly-typed directory is "entered": the shared search lists its
	// children too, so path navigation does not require a trailing slash.
	const search = useDebouncedSearch({
		canRun: () => isOpen && fileSearchEnabled,
		debounceMs: SEARCH_DEBOUNCE_MS,
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
				// Generous limit: ranking is client-side, only the top
				// MAX_RESULTS_SHOWN are shown.
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
					nav.bumpScroll(); // scroll the list back to the top (first item is hovered)
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
	// Single funnel for every local close so the host refocus always fires.
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

	// Resolve a browser-picked folder name (which exposes only the leaf name)
	// to a server-side absolute path; null when the server cannot locate it,
	// so the caller fails visibly instead of committing a bare leaf name.
	async function resolveNativeName(name: string): Promise<string | null> {
		try {
			const res = await ToolsService.executeToolRaw(BuiltInTool.FILE_GLOB_SEARCH, {
				include: buildCaseInsensitiveGlob(name),
				limit: NATIVE_LIMIT,
				max_depth: NATIVE_MAX_DEPTH,
				path: homeBase ?? HOME_TILDE,
				type: GlobSearchType.DIR
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

			{#if !fileSearchEnabled}
				<div class="px-2 py-1.5 text-sm text-muted-foreground">{searchUnavailableMessage}</div>
			{:else if query.trim() && (search.isSearching || queryResults.length > 0 || searchError)}
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

			{#if pickerSupported && fileSearchEnabled}
				<button
					type="button"
					class="-mt-1 flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none hover:bg-accent hover:text-accent-foreground"
					onclick={browseNative}
				>
					<FolderOpen class="size-4 shrink-0 text-muted-foreground" />
					<span>Browse</span>
				</button>
			{/if}

			{#if homeBase && fileSearchEnabled}
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
