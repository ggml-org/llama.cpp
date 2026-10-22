<script lang="ts">
	import ChatTabsItem from './ChatTabsItem.svelte';
	import ChatTabsNewChatButton from './ChatTabsNewChatButton.svelte';
	import { SquarePen } from '@lucide/svelte';
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
	import { resolveTabView } from '$lib/utils';
	import { tick } from 'svelte';

	const carousel = useScrollCarousel();

	let view = $derived(resolveTabView(page, tabsStore.openTabs));

	let activeId = $derived(view.activeId);

	let tabs = $derived(
		view.ids.map((id) =>
			id === NEW_CHAT_TAB_ID
				? { href: view.hrefFor(id), icon: SquarePen, id, name: NEW_CHAT_LABEL }
				: {
						href: view.hrefFor(id),
						id,
						name:
							conversationsStore.conversations.find((c) => c.id === id)?.name ?? UNNAMED_CHAT_LABEL
					}
		)
	);

	// hide the New chat button while a new-chat tab is already open
	let showNewChatButton = $derived(!view.ids.includes(NEW_CHAT_TAB_ID));

	let loadingIds = $derived(new Set(chatStore.getAllLoadingChats()));

	function handleClose(id: string) {
		void tabsStore.close(id, id === activeId, view);
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
	class={[
		'group sticky pl-1 top-0 z-10 hidden md:block chat-top-fade transition-[padding] duration-200 ease-in-out pt-3.25',
		uiStore.isSidebarExpanded
			? CHAT_TABS_MAX_WIDTH.EXPANDED_SIDEBAR
			: CHAT_TABS_MAX_WIDTH.COLLAPSED_SIDEBAR
	]}
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
				<ChatTabsNewChatButton
					onclick={() => void tabsStore.openInNewTab(NEW_CHAT_TAB_ID, { activate: true })}
				/>
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
