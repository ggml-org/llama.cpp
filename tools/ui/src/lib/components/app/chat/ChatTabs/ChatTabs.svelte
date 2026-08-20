<script lang="ts">
	import { Loader2, Plus, SquarePen, X } from '@lucide/svelte';
	import { page } from '$app/state';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';
	import { chatStore, conversationsStore, NEW_CHAT_TAB_ID, tabsStore } from '$lib/stores';

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
</script>

<nav
	class="sticky pl-1 top-0 pt-3.5 z-10 hidden md:block transition-colors ease-in-out"
	aria-label="Open conversations"
>
	<div class="flex h-10 items-center gap-1.25 overflow-x-auto px-2">
		{#each tabs as tab (tab.id)}
			{@const isActive = tab.id === activeId}
			{@const isLoading = loadingIds.has(tab.id)}

			<div
				class={cn(
					'group flex h-8 max-w-52 min-w-0 shrink-0 items-center gap-1 rounded-lg pr-1 pl-3 text-sm transition-colors backdrop-blur-xl',
					isActive
						? 'bg-foreground/8 text-foreground'
						: 'text-muted-foreground hover:bg-foreground/5 hover:text-foreground'
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
							class="backdrop-blur-lg flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-foreground/5 hover:text-foreground"
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
</nav>
