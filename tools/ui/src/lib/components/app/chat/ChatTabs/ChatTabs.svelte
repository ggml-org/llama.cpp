<script lang="ts">
	import { ChevronLeft, ChevronRight, Loader2, Plus, SquarePen, X } from '@lucide/svelte';
	import { page } from '$app/state';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';
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
	class="sticky pl-1 top-0 z-10 hidden md:block transition-[padding] duration-200 ease-in-out pt-3.25 {uiStore.isSidebarExpanded
		? 'pt-1.5 max-w-[calc(100vw-19.5rem)]'
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
					{@const isActive = tab.id === activeId}
					{@const isLoading = loadingIds.has(tab.id)}

					<div
						data-active-tab={isActive ? 'true' : undefined}
						class={cn(
							'group flex h-8 max-w-52 min-w-0 shrink-0 items-center gap-1 rounded-lg pr-1 pl-3 text-sm whitespace-nowrap transition-colors hover:bg-foreground/10 border backdrop-blur-xl first:ml-2',
							isActive
								? 'bg-muted/60 border-border/10 shadow-sm text-accent-foreground hover:bg-primary/15'
								: 'text-muted-foreground hover:text-foreground border-transparent hover:bg-primary/10 hover:border-border/10 hover:shadow-sm'
						)}
					>
						<button
							class="flex min-w-0 flex-1 cursor-pointer items-center gap-2"
							onclick={() => tabsStore.activate(tab.id)}
							onauxclick={(e) => handleAuxClick(tab.id, e)}
							aria-current={isActive ? 'page' : undefined}
						>
							{#if isLoading}
								<Loader2 class="h-3.5 w-3.5 shrink-0 animate-spin" />
							{:else if tab.isNewChat}
								<SquarePen class="h-3.5 w-3.5 shrink-0" />
							{/if}

							<span class="truncate">{tab.name}</span>
						</button>

						<Tooltip.Root>
							<Tooltip.Trigger>
								{#snippet child({ props })}
									<button
										{...props}
										class={cn(
											'flex h-5 w-5 shrink-0 cursor-pointer items-center justify-center rounded-sm text-muted-foreground transition-opacity hover:bg-foreground/10 hover:text-foreground'
										)}
										onclick={() => handleClose(tab.id)}
										aria-label="Close tab"
									>
										<X class="h-3.5 w-3.5" />
									</button>
								{/snippet}
							</Tooltip.Trigger>

							<Tooltip.Content>
								<p>Close tab</p>
							</Tooltip.Content>
						</Tooltip.Root>
					</div>
				{/each}

				{#if showNewChatButton}
					<Tooltip.Root>
						<Tooltip.Trigger>
							{#snippet child({ props })}
								<button
									{...props}
									class="backdrop-blur-lg flex h-8 w-8 mr-4 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-foreground/5 hover:text-foreground"
									onclick={() => conversationsStore.openNewChat()}
									aria-label="New chat"
								>
									<Plus class="h-4 w-4" />
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
