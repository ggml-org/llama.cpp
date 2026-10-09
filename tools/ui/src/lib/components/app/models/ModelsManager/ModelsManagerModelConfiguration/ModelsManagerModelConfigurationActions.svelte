<script lang="ts">
	import { Eject, Loader2, MessageSquare, Power, SquarePen } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { conversationsStore, deviceStore } from '$lib/stores';

	interface Props {
		/** A provider that cannot load or unload has no button to offer. */
		canToggleLoad?: boolean;
		isLoaded: boolean;
		isLoading: boolean;
		onToggleLoad: () => void;
		onUseInChat: () => void;
		onUseInNewChat: () => void;
		/** The header gives the actions room to breathe, the phone bar keeps them small. */
		size?: 'sm' | 'default';
	}

	let {
		canToggleLoad = true,
		isLoaded,
		isLoading,
		onToggleLoad,
		onUseInChat,
		onUseInNewChat,
		size = 'sm'
	}: Props = $props();

	// a chat with anything in it takes the model, a new chat is not what its user asked for
	let hasChat = $derived(
		conversationsStore.activeConversation !== null ||
			(conversationsStore.activeMessages as DatabaseMessage[]).length > 0
	);
	let isMobile = $derived(deviceStore.isMobile);
</script>

<div class="my-3 flex flex-col gap-2">
	{#if isMobile}
		<!-- a phone picks its model in the manager, so pointing this chat at it leads -->
		<Button class="w-full gap-1.5" onclick={hasChat ? onUseInChat : onUseInNewChat} {size}>
			{#if hasChat}
				<MessageSquare class="h-3.5 w-3.5" />

				Use in this chat
			{:else}
				<SquarePen class="h-3.5 w-3.5" />

				Start a new chat
			{/if}
		</Button>
	{/if}

	<div class="flex gap-3 mt-2">
		{#if !isMobile}
			<Button class="flex-1 gap-1.5" onclick={onUseInNewChat} {size}>
				<SquarePen class="h-3.5 w-3.5" />

				Start a new chat
			</Button>
		{/if}

		{#if canToggleLoad}
			<Button
				class="flex-1 gap-1.5"
				disabled={isLoading}
				onclick={onToggleLoad}
				{size}
				variant="secondary"
			>
				{#if isLoading}
					<Loader2 class="h-3.5 w-3.5 animate-spin" />

					Loading...
				{:else if isLoaded}
					<Eject class="h-3.5 w-3.5" />

					Unload model
				{:else}
					<Power class="h-3.5 w-3.5" />

					Load model
				{/if}
			</Button>
		{/if}
	</div>
</div>
