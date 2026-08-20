<script lang="ts">
	import ChatTabItem from './ChatTabItem.svelte';
	import { ChevronLeft, ChevronRight, Plus } from '@lucide/svelte';
	import { page } from '$app/state';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { useScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';
	import { chatStore, conversationsStore, NEW_CHAT_TAB_ID, tabsStore, uiStore } from '$lib/stores';

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

	// hide the New chat button when the new-chat tab is the only one open
	let showNewChatButton = $derived(tabsStore.openTabs.some((id) => id !== NEW_CHAT_TAB_ID));

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

	$effect(() => {
		// keep the active tab in view whenever it changes (activeId drives re-run)
		const el = carousel.scrollContainer?.querySelector<HTMLElement>('[data-active-tab]');

		if (el && activeId) {
			carousel.scrollToCenter(el);
		}
	});
</script>

<nav
	class="group sticky pl-1 top-0 z-10 hidden md:block chat-tabs-fade transition-[padding] duration-200 ease-in-out pt-3.25 {uiStore.isSidebarExpanded
		? 'max-w-[calc(100vw-19.5rem)]'
		: 'max-w-[calc(100vw-4.5rem)]'}"
	aria-label="Open conversations"
>
	<div class="relative flex h-10 items-center" style="scroll-padding: 1rem;">
		<button
			class="absolute left-2 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-muted shadow-md backdrop-blur-sm transition-opacity hover:bg-accent {carousel.canScrollLeft
				? 'opacity-100'
				: 'pointer-events-none opacity-0'}"
			onclick={carousel.scrollLeft}
			aria-label="Scroll left"
		>
			<ChevronLeft class={ICON_CLASS_DEFAULT} />
		</button>

		<div
			class="scrollbar-hide flex h-10 min-w-0 items-center overflow-x-auto"
			bind:this={carousel.scrollContainer}
			onscroll={carousel.updateScrollButtons}
		>
			<div class="flex min-w-max items-center gap-1.25">
				{#each tabs as tab (tab.id)}
					<ChatTabItem
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
			</div>
		</div>

		<button
			class="absolute right-2 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-muted shadow-md backdrop-blur-sm transition-opacity hover:bg-accent {carousel.canScrollRight
				? 'opacity-100'
				: 'pointer-events-none opacity-0'}"
			onclick={carousel.scrollRight}
			aria-label="Scroll right"
		>
			<ChevronRight class={ICON_CLASS_DEFAULT} />
		</button>
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
