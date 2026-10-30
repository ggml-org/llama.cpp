<script lang="ts">
	import {
		Compass,
		Eject,
		Loader2,
		MessageSquare,
		Pause,
		Play,
		Power,
		SquarePen,
		Trash2
	} from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { conversationsStore, deviceStore, uiStore } from '$lib/stores';

	interface Props {
		/** A provider that cannot load or unload has no button to offer. */
		canToggleLoad?: boolean;
		isDownload?: boolean;
		isDownloadPaused?: boolean;
		isLoaded: boolean;
		isLoading: boolean;
		/** Opens the manager dialog on the Discover view, focused on this model's repo. */
		onSeeInDiscover?: () => void;
		/** Confirm-cancel dialog on the pane, so nothing deletes without a click. */
		onRequestCancelDownload?: () => void;
		onToggleDownload?: () => void;
		onToggleLoad: () => void;
		onUseInChat: () => void;
		onUseInNewChat: () => void;
		/** The header gives the actions room to breathe, the phone bar keeps them small. */
		size?: 'sm' | 'default';
	}

	let {
		canToggleLoad = true,
		isDownload = false,
		isDownloadPaused = false,
		isLoaded,
		isLoading,
		onRequestCancelDownload,
		onSeeInDiscover,
		onToggleDownload,
		onToggleLoad,
		onUseInChat,
		onUseInNewChat,
		size = 'sm'
	}: Props = $props();

	let discoverEnabled = $derived(uiStore.isDiscoverEnabled);

	// a chat with anything in it takes the model, a new chat is not what its user asked for
	let hasChat = $derived(
		conversationsStore.activeConversation !== null ||
			(conversationsStore.activeMessages as DatabaseMessage[]).length > 0
	);
	let isMobile = $derived(deviceStore.isMobile);
</script>

<div class="my-3 flex flex-col gap-2">
	{#if isDownload}
		<!-- an in-flight download cannot load or chat: it pauses, resumes, cancels -->
		<div class="flex gap-3">
			<Button class="flex-1 gap-1.5" onclick={onToggleDownload} {size} variant="outline">
				{#if isDownloadPaused}
					<Play class="size-icon-md" />

					Resume downloading
				{:else}
					<Pause class="size-icon-md" />

					Pause downloading
				{/if}
			</Button>

			<Button class="flex-1 gap-1.5" onclick={onRequestCancelDownload} {size} variant="outline">
				<Trash2 class="size-icon-md" />

				Cancel download
			</Button>
		</div>

		{#if onSeeInDiscover && discoverEnabled}
			<Button class="gap-1.5" onclick={onSeeInDiscover} {size} variant="secondary">
				<Compass class="size-icon-md" />

				See in Discover
			</Button>
		{/if}
	{:else}
		{#if isMobile}
			<!-- a phone picks its model in the manager, so pointing this chat at it leads -->
			<Button class="w-full gap-1.5" onclick={hasChat ? onUseInChat : onUseInNewChat} {size}>
				{#if hasChat}
					<MessageSquare class="size-icon-md" />

					Use in this chat
				{:else}
					<SquarePen class="size-icon-md" />

					Start a new chat
				{/if}
			</Button>
		{/if}

		<div class="flex gap-3 mt-2">
			{#if !isMobile}
				<Button class="flex-1 gap-1.5" onclick={onUseInNewChat} {size}>
					<SquarePen class="size-icon-md" />

					Start a new chat
				</Button>
			{/if}

			{#if canToggleLoad}
				<Button
					class="flex-1 gap-1.5"
					disabled={isLoading}
					onclick={onToggleLoad}
					{size}
					variant="outline"
				>
					{#if isLoading}
						<Loader2 class="size-icon-md animate-spin" />

						Loading...
					{:else if isLoaded}
						<Eject class="size-icon-md" />

						Unload model
					{:else}
						<Power class="size-icon-md" />

						Load model
					{/if}
				</Button>
			{/if}

			{#if onSeeInDiscover && discoverEnabled}
				<Button class="flex-1 gap-1.5" onclick={onSeeInDiscover} {size} variant="secondary">
					<Compass class="size-icon-md" />

					See in Discover
				</Button>
			{/if}
		</div>
	{/if}
</div>
