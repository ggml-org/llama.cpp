<script lang="ts">
	import * as Dialog from '@ggml-org/llama-ui/ui/dialog';
	import {
		ArrowDown,
		ArrowUp,
		CornerDownLeft,
		Loader2,
		MessageSquare,
		Pin,
		X
	} from '@lucide/svelte';
	import { ActionIcon, KeyboardShortcutInfo, ModelOrgAvatar } from '#lib/components/app/index.ts';
	import {
		CONVERSATION_SEARCH_CONTENT_DEBOUNCE_MS,
		CONVERSATION_SEARCH_CONTENT_LIMIT,
		CONVERSATION_SEARCH_HUB_DEBOUNCE_MS,
		CONVERSATION_SEARCH_HUB_LIMIT,
		CONVERSATION_SEARCH_MATCH_LIMIT,
		CONVERSATION_SEARCH_RECENT_LIMIT,
		DISCOVER_ICON,
		ICON_CLASS_DEFAULT,
		ICON_CLASS_SM,
		ICON_CLASS_XS,
		SIDEBAR_ACTIONS_ITEMS,
		sidebarFooterItems
	} from '#lib/constants/index.ts';
	import { KeyboardKey, SidebarAction } from '#lib/enums/index.ts';
	import { useDebouncedSearch } from '#lib/hooks/use-debounced-search.svelte.ts';
	import {
		type ConversationContentMatch,
		ConversationSearchService,
		HuggingFaceService
	} from '#lib/services/index.ts';
	import { RouterService } from '#lib/services/router.service.ts';
	import { agenticMode } from '#lib/stores/agentic-mode.svelte.ts';
	import {
		conversationsStore,
		deviceStore,
		modelsDiscoverStore,
		uiStore
	} from '#lib/stores/index.ts';
	import type { DatabaseConversation } from '#lib/types/database.js';
	import type { HfModelInfo } from '#lib/types/huggingface.js';
	import type { ConversationSearchSnippet } from '#lib/utils/conversation-search.ts';
	import { formatRelativeTime, isIMEComposing, orgOf } from '#lib/utils/index.ts';
	import { goto } from '$app/navigation';
	import { type Component, untrack } from 'svelte';

	interface Props {
		open: boolean;
		/** Runs a sidebar action picked from the actions section. */
		onAction?: (action: SidebarAction) => void;
	}

	let { onAction, open = $bindable() }: Props = $props();

	interface SearchEntry {
		icon: Component;
		id: string;
		keys?: string[];
		label: string;
		meta?: string;
		/** Org whose avatar stands in for the icon, e.g. a Hub model's base org. */
		org?: string;
		pinned?: boolean;
		/** Where the query matched, shown under the label */
		snippet?: ConversationSearchSnippet;
		run: () => void;
	}

	interface SearchSection {
		entries: SearchEntry[];
		/** Results still on their way; the section shows while they load. */
		loading?: boolean;
		title: string;
	}

	let query = $state('');
	let highlighted = $state(0);
	let inputRef = $state<HTMLInputElement | null>(null);

	const needle = $derived(query.trim().toLowerCase());
	const limit = $derived(
		needle ? CONVERSATION_SEARCH_MATCH_LIMIT : CONVERSATION_SEARCH_RECENT_LIMIT
	);

	function matches(label: string): boolean {
		return !needle || label.toLowerCase().includes(needle);
	}

	function chatEntry(
		conversation: DatabaseConversation,
		snippet?: ConversationSearchSnippet
	): SearchEntry {
		return {
			icon: MessageSquare,
			id: `chat-${conversation.id}`,
			label: conversation.name,
			meta: formatRelativeTime(conversation.lastModified),
			pinned: conversation.pinned,
			run: () => close(() => goto(RouterService.chat(conversation.id))),
			snippet
		};
	}

	const recentConversations = $derived(
		[...conversationsStore.conversations].sort((a, b) => b.lastModified - a.lastModified)
	);

	const titleMatches = $derived(
		recentConversations.filter((conversation) => matches(conversation.name))
	);

	const chatEntries = $derived(
		titleMatches.slice(0, limit).map((conversation) => chatEntry(conversation))
	);

	// chats found by what was said in them, not by their title; scanned once typing settles
	let contentMatches = $state.raw<ConversationContentMatch[]>([]);

	const contentSearch = useDebouncedSearch({
		canRun: () => open,
		debounceMs: CONVERSATION_SEARCH_CONTENT_DEBOUNCE_MS,
		getQuery: () => query.trim(),
		run: async (term, _signal, isCurrent) => {
			// a chat already listed by its title is not scanned again
			const byTitle = new Set(titleMatches.map((conversation) => conversation.id));

			try {
				const found = await ConversationSearchService.searchContent(
					recentConversations.filter((conversation) => !byTitle.has(conversation.id)),
					term,
					{ isCurrent, limit: CONVERSATION_SEARCH_CONTENT_LIMIT }
				);

				if (isCurrent()) contentMatches = found;
			} catch {
				if (isCurrent()) contentMatches = [];
			}
		}
	});

	const contentEntries = $derived(
		needle
			? contentMatches.map(({ conversation, snippet }) => ({
					...chatEntry(conversation, snippet),
					id: `message-${conversation.id}`
				}))
			: []
	);

	// Discover data: the catalog suggestions, and Hugging Face matches once typing settles
	const isDiscoverEnabled = $derived(uiStore.isDiscoverEnabled);

	// the catalog loads once a session; opening the search asks for it if it has not yet.
	// untracked: fetch() toggles its loading flag synchronously
	$effect(() => {
		if (open && isDiscoverEnabled) untrack(() => void modelsDiscoverStore.fetch());
	});

	/** A Hub repo as a result; picking it opens Discover on that repo. */
	function modelEntry(model: HfModelInfo, section: string): SearchEntry {
		const range = modelsDiscoverStore.cachedSizeRangeFor(model.id);

		return {
			icon: DISCOVER_ICON,
			id: `${section}-${model.id}`,
			label: model.id,
			meta: range
				? HuggingFaceService.formatSizeRange(range.min, range.max)
				: `${HuggingFaceService.formatDownloads(model.downloads)} downloads`,
			// the model's own org, the way the model cards show it
			org: orgOf(HuggingFaceService.getBaseModels(model)[0]) || orgOf(model.id),
			run: () => close(() => uiStore.openModelsDiscover(model.id))
		};
	}

	const suggestedEntries = $derived.by((): SearchEntry[] =>
		isDiscoverEnabled && needle
			? modelsDiscoverStore.catalogModels
					.filter((model) => matches(model.id))
					.slice(0, limit)
					.map((model) => modelEntry(model, 'suggested'))
			: []
	);

	// the dialog keeps its own results: the Discover search stays as it was
	let hubResults = $state.raw<HfModelInfo[]>([]);

	const hubSearch = useDebouncedSearch({
		canRun: () => open && isDiscoverEnabled,
		debounceMs: CONVERSATION_SEARCH_HUB_DEBOUNCE_MS,
		getQuery: () => query.trim(),
		run: async (term, _signal, isCurrent) => {
			try {
				const results = await HuggingFaceService.searchByQuery(term, {
					limit: CONVERSATION_SEARCH_HUB_LIMIT
				});

				if (isCurrent()) hubResults = modelsDiscoverStore.baseOrgFirst(results);
			} catch {
				// offline or rate limited: the local results stand on their own
				if (isCurrent()) hubResults = [];
			}
		}
	});

	$effect(() => {
		const term = query.trim();
		const isOpen = open;
		const withHub = isDiscoverEnabled;

		untrack(() => {
			if (!isOpen || !term) {
				contentSearch.cancel();
				hubSearch.cancel();
				contentMatches = [];
				hubResults = [];

				return;
			}

			contentSearch.setLoading(true);
			contentSearch.run(term);

			if (!withHub) {
				hubSearch.cancel();
				hubResults = [];

				return;
			}

			hubSearch.setLoading(true);
			hubSearch.run(term);
		});
	});

	// a suggestion that matches is listed once, under the suggestions
	const hubEntries = $derived.by((): SearchEntry[] => {
		if (!needle) return [];

		const suggested = new Set(suggestedEntries.map((entry) => entry.label));

		return hubResults
			.filter((model) => !suggested.has(model.id))
			.map((model) => modelEntry(model, 'hub'));
	});

	const actionEntries = $derived.by((): SearchEntry[] =>
		// every sidebar action the sidebar shows but Search itself, which is what opened this
		// dialog, and the models manager on a phone, which picks its models in the selector
		[...SIDEBAR_ACTIONS_ITEMS, ...sidebarFooterItems(agenticMode.enabled)]
			.filter(
				(item) =>
					item.action &&
					item.action !== SidebarAction.SEARCH &&
					uiStore.isSidebarActionShown(item.action) &&
					(item.action !== SidebarAction.MANAGE_MODELS || !deviceStore.isMobile)
			)
			.map((item) => ({
				icon: item.icon,
				id: `action-${item.action}`,
				keys: item.keys,
				label: item.tooltip,
				run: () => close(() => onAction?.(item.action!))
			}))
			.filter((entry) => matches(entry.label))
	);

	// your own chats first; the Hub answers last, so it comes last and never moves a row above it
	const sections = $derived<SearchSection[]>(
		[
			{ entries: chatEntries, title: needle ? 'Chats' : 'Recents' },
			{
				entries: contentEntries,
				loading: contentSearch.isSearching && !!needle,
				title: 'In messages'
			},
			{ entries: suggestedEntries, title: 'Suggested models' },
			{ entries: actionEntries, title: 'Actions' },
			{
				entries: hubEntries,
				loading: hubSearch.isSearching && !!needle,
				title: 'On Hugging Face'
			}
		].filter((section) => section.entries.length > 0 || section.loading)
	);

	const entries = $derived(sections.flatMap((section) => section.entries));
	const activeIndex = $derived(Math.min(highlighted, entries.length - 1));

	const emptyMessage = $derived(needle ? 'No results found' : 'No chats yet');

	/** Closes the search first, so a dialog opened by `next` takes the focus. */
	function close(next: () => void) {
		handleOpenChange(false);
		next();
	}

	function move(step: number) {
		if (entries.length === 0) return;

		highlighted = (activeIndex + step + entries.length) % entries.length;
		document.getElementById(`search-result-${highlighted}`)?.scrollIntoView({ block: 'nearest' });
	}

	function handleKeydown(event: KeyboardEvent) {
		// while an IME is composing, Enter confirms a candidate and the arrows move through them
		if (isIMEComposing(event)) return;

		if (event.key === KeyboardKey.ARROW_DOWN) {
			event.preventDefault();
			move(1);
		} else if (event.key === KeyboardKey.ARROW_UP) {
			event.preventDefault();
			move(-1);
		} else if (event.key === KeyboardKey.ENTER) {
			event.preventDefault();
			entries[activeIndex]?.run();
		}
	}

	function handleOpenChange(nextOpen: boolean) {
		if (!nextOpen) {
			query = '';
			highlighted = 0;
		}

		open = nextOpen;
	}
</script>

{#snippet hint(label: string, keys: (string | Component)[])}
	<span class="flex items-center gap-1.5">
		{label}

		<span class="flex items-center gap-1">
			{#each keys as Key, index (index)}
				<kbd
					class="inline-flex h-5 min-w-5 items-center justify-center rounded border border-border/60 px-1 font-sans text-caption"
				>
					{#if typeof Key === 'string'}
						{Key}
					{:else}
						<Key class={ICON_CLASS_XS} />
					{/if}
				</kbd>
			{/each}
		</span>
	</span>
{/snippet}

{#snippet search()}
	<div class="border-b border-border/30 px-4 py-3 dark:border-border/20">
		<div class="flex items-center gap-2">
			<input
				bind:this={inputRef}
				bind:value={query}
				aria-activedescendant={entries.length > 0 ? `search-result-${activeIndex}` : undefined}
				aria-controls="search-results"
				aria-expanded="true"
				autocomplete="off"
				class="h-10 min-w-0 flex-1 border-0 bg-transparent px-1 text-title font-normal outline-none placeholder:text-muted-foreground focus:outline-none focus-visible:ring-0"
				oninput={() => (highlighted = 0)}
				onkeydown={handleKeydown}
				placeholder="Search"
				role="combobox"
				spellcheck={false}
			/>

			<ActionIcon
				ariaLabel="Close search"
				class="size-8 shrink-0 text-muted-foreground"
				icon={X}
				iconSize={ICON_CLASS_DEFAULT}
				onclick={() => handleOpenChange(false)}
			/>
		</div>
	</div>

	<div class="min-h-0 flex-1 overflow-y-auto p-2" id="search-results" role="listbox">
		{#each sections as section, sectionIndex (section.title)}
			{@const offset = sections
				.slice(0, sectionIndex)
				.reduce((total, previous) => total + previous.entries.length, 0)}

			<div class="flex flex-col gap-0.5 {sectionIndex > 0 ? 'mt-3' : ''}" role="group">
				<p class="flex items-center gap-1.5 px-2.5 pt-1 pb-1.5 text-label-sm text-muted-foreground">
					{section.title}

					{#if section.loading}
						<Loader2 aria-label="Searching" class="{ICON_CLASS_XS} animate-spin" />
					{/if}
				</p>

				{#each section.entries as entry, entryIndex (entry.id)}
					{@const index = offset + entryIndex}
					{@const isActive = index === activeIndex && !deviceStore.isMobile}

					<!-- a phone has no keyboard to move a highlight with: no row is lit there -->
					<button
						aria-selected={isActive}
						class="flex w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-body-sm transition-colors max-md:rounded-xl max-md:py-3 max-md:text-body {isActive
							? 'bg-accent text-accent-foreground'
							: ''}"
						id="search-result-{index}"
						onclick={entry.run}
						onmousemove={() => (highlighted = index)}
						role="option"
						tabindex={-1}
						type="button"
					>
						{#if entry.org && HuggingFaceService.isEnabled()}
							<ModelOrgAvatar class="shrink-0" org={entry.org} size="size-icon-md" />
						{:else}
							<entry.icon class="{ICON_CLASS_DEFAULT} shrink-0 text-muted-foreground" />
						{/if}

						{#if entry.snippet}
							<span class="flex min-w-0 flex-1 flex-col">
								<span class="truncate">{entry.label}</span>

								<span class="truncate text-caption text-muted-foreground">
									{entry.snippet.before}<mark class="bg-transparent font-medium text-foreground"
										>{entry.snippet.match}</mark
									>{entry.snippet.after}
								</span>
							</span>
						{:else}
							<span class="min-w-0 flex-1 truncate">{entry.label}</span>
						{/if}

						{#if entry.pinned}
							<Pin aria-label="Pinned" class="{ICON_CLASS_SM} shrink-0 text-muted-foreground" />
						{/if}

						{#if entry.meta}
							<span class="max-w-2/5 shrink-0 truncate text-caption text-muted-foreground">
								{entry.meta}
							</span>
						{/if}

						{#if entry.keys && !deviceStore.isMobile}
							<KeyboardShortcutInfo class="opacity-100" keys={entry.keys} />
						{/if}

						<CornerDownLeft
							class="{ICON_CLASS_SM} shrink-0 max-md:hidden text-muted-foreground {isActive
								? 'opacity-100'
								: 'opacity-0'}"
						/>
					</button>
				{/each}
			</div>
		{:else}
			<p class="px-3 py-12 text-center text-body-sm text-muted-foreground">{emptyMessage}</p>
		{/each}
	</div>

	<div
		class="flex items-center gap-5 border-t border-border/30 px-4 py-2.5 text-caption text-muted-foreground max-md:hidden dark:border-border/20"
	>
		{@render hint('Close', ['Esc'])}

		{@render hint('Navigate', [ArrowUp, ArrowDown])}

		<span class="ml-auto">{@render hint('Open', [CornerDownLeft])}</span>
	</div>
{/snippet}

<!-- like every dialog, a phone shows it full screen -->
<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="top-[12%] flex h-[min(36rem,76dvh)]! w-full max-w-2xl! md:w-[calc(100%-2rem)] translate-y-0 flex-col gap-0 overflow-hidden rounded-xl p-0"
		onOpenAutoFocus={(event) => {
			event.preventDefault();
			inputRef?.focus();
		}}
	>
		<Dialog.Title class="sr-only">Search</Dialog.Title>

		{@render search()}
	</Dialog.Content>
</Dialog.Root>
