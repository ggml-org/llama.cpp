<script lang="ts">
	import ChatTabsItem from './ChatTabsItem.svelte';
	import { Plus } from '@lucide/svelte';
	import { page } from '$app/state';
	import { ScrollCarousel } from '$lib/components/ui/scroll-carousel';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { useScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';
	import { chatStore, conversationsStore, NEW_CHAT_TAB_ID, tabsStore, uiStore } from '$lib/stores';
	import { tick } from 'svelte';

	const carousel = useScrollCarousel();

	let activeId = $derived(page.params.id ?? NEW_CHAT_TAB_ID);

	let tabs = $derived(
		tabsStore.openTabs.map((id) => ({
			id,
			isNewChat: id === NEW_CHAT_TAB_ID,
			name:
				id === NEW_CHAT_TAB_ID
					? 'New chat'
					: (conversationsStore.conversations.find((c) => c.id === id)?.name ?? 'Chat')
		}))
	);

	// hide the New chat button while a new-chat tab is already open
	let showNewChatButton = $derived(!tabsStore.openTabs.includes(NEW_CHAT_TAB_ID));

	let loadingIds = $derived(new Set(chatStore.getAllLoadingChats()));

	function handleClose(id: string) {
		void tabsStore.close(id, activeId ?? null);
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
		const addedIds = tabs.filter((t) => !previousTabIds.has(t.id)).map((t) => t.id);

		previousTabIds = currentIds;

		const activeChanged = activeId !== previousActiveId;

		previousActiveId = activeId;

		// scroll when the active tab changes (a click) or when a new tab is added
		if (addedIds.length === 0 && !activeChanged) return;

		// wait for the new tab to be laid out before scrolling to it
		void tick().then(() => {
			const el = carousel.scrollContainer?.querySelector<HTMLElement>('[data-active-tab]');

			if (el && activeId) {
				carousel.scrollToCenter(el);
			}
		});
	});
</script>

<nav
	class="group sticky pl-1 top-0 z-10 hidden md:block chat-tabs-fade transition-[padding] duration-200 ease-in-out pt-3.25 {uiStore.isSidebarExpanded
		? 'max-w-[calc(100vw-19.5rem)]'
		: 'max-w-[calc(100vw-4.5rem)]'}"
	aria-label="Open conversations"
>
	<div class="relative">
		<ScrollCarousel
			class="h-10"
			containerClass="flex h-10 min-w-0 items-center"
			innerClass="items-center gap-1.25"
			{carousel}
		>
			{#each tabs as tab (tab.id)}
				<ChatTabsItem
					{tab}
					isActive={tab.id === activeId}
					isLoading={loadingIds.has(tab.id)}
					onActivate={(id) => tabsStore.activate(id)}
					onClose={handleClose}
					onStop={handleStop}
					onAuxClick={handleAuxClick}
				/>
			{/each}

			{#if showNewChatButton}
				<Tooltip.Root>
					<Tooltip.Trigger>
						{#snippet child({ props })}
							<button
								{...props}
								class="backdrop-blur-lg flex h-8 w-8 mr-4 shrink-0 cursor-pointer items-center justify-center rounded-md transition-colors hover:bg-foreground/5"
								onclick={() => conversationsStore.openNewChat()}
								aria-label="New chat"
							>
								<Plus class="h-4 w-4 opacity-40 transition-opacity group-hover:opacity-100" />
							</button>
						{/snippet}
					</Tooltip.Trigger>

					<Tooltip.Content>
						<p>New chat</p>
					</Tooltip.Content>
				</Tooltip.Root>
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
