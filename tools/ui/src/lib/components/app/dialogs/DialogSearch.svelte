<script lang="ts">
	import { ArrowDown, ArrowUp, CornerDownLeft, MessageSquare, Pin, X } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { ActionIcon, KeyboardShortcutInfo } from '$lib/components/app';
	import * as Dialog from '$lib/components/ui/dialog';
	import {
		ICON_CLASS_DEFAULT,
		ICON_CLASS_SM,
		ICON_CLASS_XS,
		SIDEBAR_ACTIONS_ITEMS
	} from '$lib/constants';
	import { KeyboardKey, SidebarAction } from '$lib/enums';
	import { RouterService } from '$lib/services/router.service';
	import { conversationsStore } from '$lib/stores';
	import { formatRelativeTime, isIMEComposing } from '$lib/utils';
	import type { Component } from 'svelte';

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
		pinned?: boolean;
		run: () => void;
	}

	interface SearchSection {
		entries: SearchEntry[];
		title: string;
	}

	/** Recent chats shown before typing, so the actions stay in view below them. */
	const RECENT_LIMIT = 5;
	/** Matches kept while typing; the rest stays reachable by typing more. */
	const MATCH_LIMIT = 50;

	let query = $state('');
	let highlighted = $state(0);
	let inputRef = $state<HTMLInputElement | null>(null);

	const needle = $derived(query.trim().toLowerCase());
	const limit = $derived(needle ? MATCH_LIMIT : RECENT_LIMIT);

	function matches(label: string): boolean {
		return !needle || label.toLowerCase().includes(needle);
	}

	const chatEntries = $derived.by((): SearchEntry[] =>
		conversationsStore.conversations
			.filter((conversation) => matches(conversation.name))
			.sort((a, b) => b.lastModified - a.lastModified)
			.slice(0, limit)
			.map((conversation) => ({
				icon: MessageSquare,
				id: `chat-${conversation.id}`,
				label: conversation.name,
				meta: formatRelativeTime(conversation.lastModified),
				pinned: conversation.pinned,
				run: () => close(() => goto(RouterService.chat(conversation.id)))
			}))
	);

	const actionEntries = $derived.by((): SearchEntry[] =>
		// every sidebar action but Search itself, which is what opened this dialog
		SIDEBAR_ACTIONS_ITEMS.filter((item) => item.action && item.action !== SidebarAction.SEARCH)
			.map((item) => ({
				icon: item.icon,
				id: `action-${item.action}`,
				keys: item.keys,
				label: item.tooltip,
				run: () => close(() => onAction?.(item.action!))
			}))
			.filter((entry) => matches(entry.label))
	);

	const sections = $derived<SearchSection[]>(
		[
			{ entries: chatEntries, title: needle ? 'Chats' : 'Recents' },
			{ entries: actionEntries, title: 'Actions' }
		].filter((section) => section.entries.length > 0)
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
					class="inline-flex h-5 min-w-5 items-center justify-center rounded border border-border/60 px-1 font-sans text-[11px]"
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

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="top-[12%] flex h-[min(36rem,76dvh)]! w-full max-w-2xl! translate-y-0 flex-col gap-0 overflow-hidden rounded-xl p-0 max-md:top-4 md:w-full md:max-w-2xl!"
		onOpenAutoFocus={(event) => {
			event.preventDefault();
			inputRef?.focus();
		}}
	>
		<Dialog.Title class="sr-only">Search</Dialog.Title>

		<div class="border-b border-border/30 px-4 py-3 dark:border-border/20">
			<div class="flex items-center gap-2">
				<input
					bind:this={inputRef}
					bind:value={query}
					aria-activedescendant={entries.length > 0 ? `search-result-${activeIndex}` : undefined}
					aria-controls="search-results"
					aria-expanded="true"
					autocomplete="off"
					class="h-10 min-w-0 flex-1 border-0 bg-transparent px-1 text-lg outline-none placeholder:text-muted-foreground focus:outline-none focus-visible:ring-0"
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
					<p class="px-2.5 pt-1 pb-1.5 text-xs font-medium text-muted-foreground">
						{section.title}
					</p>

					{#each section.entries as entry, entryIndex (entry.id)}
						{@const index = offset + entryIndex}
						{@const isActive = index === activeIndex}

						<button
							aria-selected={isActive}
							class="flex w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors {isActive
								? 'bg-accent text-accent-foreground'
								: ''}"
							id="search-result-{index}"
							onclick={entry.run}
							onmousemove={() => (highlighted = index)}
							role="option"
							tabindex={-1}
							type="button"
						>
							<entry.icon class="{ICON_CLASS_DEFAULT} shrink-0 text-muted-foreground" />

							<span class="min-w-0 flex-1 truncate">{entry.label}</span>

							{#if entry.pinned}
								<Pin aria-label="Pinned" class="{ICON_CLASS_SM} shrink-0 text-muted-foreground" />
							{/if}

							{#if entry.meta}
								<span class="max-w-[40%] shrink-0 truncate text-xs text-muted-foreground">
									{entry.meta}
								</span>
							{/if}

							{#if entry.keys}
								<KeyboardShortcutInfo class="opacity-100" keys={entry.keys} />
							{/if}

							<CornerDownLeft
								class="{ICON_CLASS_SM} shrink-0 text-muted-foreground {isActive
									? 'opacity-100'
									: 'opacity-0'}"
							/>
						</button>
					{/each}
				</div>
			{:else}
				<p class="px-3 py-12 text-center text-sm text-muted-foreground">{emptyMessage}</p>
			{/each}
		</div>

		<div
			class="flex items-center gap-5 border-t border-border/30 px-4 py-2.5 text-xs text-muted-foreground max-md:hidden dark:border-border/20"
		>
			{@render hint('Close', ['Esc'])}

			{@render hint('Navigate', [ArrowUp, ArrowDown])}

			<span class="ml-auto">{@render hint('Open', [CornerDownLeft])}</span>
		</div>
	</Dialog.Content>
</Dialog.Root>
