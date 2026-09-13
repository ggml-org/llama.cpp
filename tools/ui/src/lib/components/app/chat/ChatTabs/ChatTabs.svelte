<script lang="ts">
	import ChatTabsItem from './ChatTabsItem.svelte';
	import ChatTabsNewChatButton from './ChatTabsNewChatButton.svelte';
	import { page } from '$app/state';
	import { ScrollCarousel } from '$lib/components/app';
	import {
		CHAT_TABS_MAX_WIDTH,
		NEW_CHAT_LABEL,
		NEW_CHAT_TAB_ID,
		UI_DATA_ATTRS,
		UNNAMED_CHAT_LABEL
	} from '$lib/constants';
	import { useScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';
	import { chatStore, conversationsStore, tabsStore, uiStore } from '$lib/stores';
	import { tick } from 'svelte';

	const carousel = useScrollCarousel();

	let activeId = $derived(page.params.id ?? NEW_CHAT_TAB_ID);

	let tabs = $derived(
		tabsStore.openTabs.map((id) => ({
			id,
			isNewChat: id === NEW_CHAT_TAB_ID,
			name:
				id === NEW_CHAT_TAB_ID
					? NEW_CHAT_LABEL
					: (conversationsStore.conversations.find((c) => c.id === id)?.name ?? UNNAMED_CHAT_LABEL)
		}))
	);

	// hide the New chat button while a new-chat tab is already open
	let showNewChatButton = $derived(!tabsStore.openTabs.includes(NEW_CHAT_TAB_ID));

	let loadingIds = $derived(new Set(chatStore.getAllLoadingChats()));

	function handleClose(id: string) {
		void tabsStore.close(id, activeId);
	}

	function handleStop(id: string, event: MouseEvent) {
		event.stopPropagation();
		void chatStore.stopGenerationForChat(id);
	}

	function handleAuxClick(id: string, event: MouseEvent) {
		// middle-click closes, like browser tabs
		if (event.button === 1) {
			event.preventDefault();
			handleClose(id);
		}
	}

	let previousTabIds = new Set<string>();
	let previousActiveId: string | null = null;

	$effect(() => {
		const currentIds = new Set(tabs.map((t) => t.id));
		const hasAddedTab = tabs.some((t) => !previousTabIds.has(t.id));

		previousTabIds = currentIds;

		const activeChanged = activeId !== previousActiveId;

		previousActiveId = activeId;

		// scroll when the active tab changes (a click) or when a new tab is added
		if (!hasAddedTab && !activeChanged) return;

		// wait for the new tab to be laid out before scrolling to it
		void tick().then(() => {
			const el = carousel.scrollContainer?.querySelector<HTMLElement>(
				`[${UI_DATA_ATTRS.ACTIVE_TAB}]`
			);

			if (el) {
				carousel.scrollToCenter(el);
			}
		});
	});
</script>

<nav
	aria-label="Open conversations"
	class="group sticky pl-1 top-0 z-10 hidden md:block chat-tabs-fade transition-[padding] duration-200 ease-in-out pt-3.25 {uiStore.isSidebarExpanded
		? CHAT_TABS_MAX_WIDTH.EXPANDED_SIDEBAR
		: CHAT_TABS_MAX_WIDTH.COLLAPSED_SIDEBAR}"
>
	<div class="relative">
		<ScrollCarousel
			{carousel}
			class="h-10"
			containerClass="flex h-10 min-w-0 items-center"
			innerClass="items-center gap-1.25"
		>
			{#each tabs as tab (tab.id)}
				<ChatTabsItem
					isActive={tab.id === activeId}
					isLoading={loadingIds.has(tab.id)}
					onActivate={(id) => tabsStore.activate(id)}
					onAuxClick={handleAuxClick}
					onClose={handleClose}
					onStop={handleStop}
					{tab}
				/>
			{/each}

			{#if showNewChatButton}
				<ChatTabsNewChatButton onclick={() => void conversationsStore.openNewChat()} />
			{/if}
		</ScrollCarousel>

		<div
			class="pointer-events-none absolute inset-y-0 left-0 z-[5] w-8 bg-gradient-to-r from-background to-transparent transition-opacity {carousel.canScrollLeft
				? 'opacity-100'
				: 'opacity-0'}"
		></div>

		<div
			class="pointer-events-none absolute inset-y-0 right-0 z-[5] w-8 bg-gradient-to-l from-background to-transparent transition-opacity {carousel.canScrollRight
				? 'opacity-100'
				: 'opacity-0'}"
		></div>
	</div>
</nav>

<style>
	.chat-tabs-fade {
		background: linear-gradient(
			to bottom,
			color-mix(in srgb, var(--background) 100%, transparent) 25%,
			color-mix(in srgb, var(--background) 80%, transparent) 50%,
			color-mix(in srgb, var(--background) 40%, transparent) 75%,
			transparent 100%
		);
	}
</style>
