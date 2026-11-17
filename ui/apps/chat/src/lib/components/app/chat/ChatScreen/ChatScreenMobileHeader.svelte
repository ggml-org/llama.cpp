<script lang="ts">
	import { HOME_TILDE, SET_WORKING_DIRECTORY_LABEL } from '$lib/constants';
	import { conversationsStore, toolsStore } from '$lib/stores';
	import { lastPathSegment } from '$lib/utils';

	interface Props {
		/** Opens the chat form's working directory picker. */
		onWorkingDirectoryClick?: () => void;
	}

	let { onWorkingDirectoryClick }: Props = $props();

	const title = $derived(conversationsStore.activeConversation?.name || 'New chat');
	const directory = $derived(
		conversationsStore.activeConversation?.cwd ?? conversationsStore.preferences.pendingCwd
	);
	const showWorkingDirectory = $derived(conversationsStore.preferences.hasEnabledCwdTools());
	// the folder's own name: the full path does not fit between the corner keys
	const directoryLabel = $derived(
		!directory
			? SET_WORKING_DIRECTORY_LABEL
			: directory === toolsStore.serverHome
				? HOME_TILDE
				: lastPathSegment(directory)
	);
</script>

<!-- a phone's chat header, between the menu key and the context gauge: the chat's title and,
     under it, its working directory, which opens the picker -->
<div
	class="sidebar-reveal-pin-top pointer-events-none fixed top-(--corner-control-top) right-16 left-16 z-20 flex h-10 flex-col items-center justify-center text-center"
>
	<p class="w-full truncate text-[15px] leading-5 font-semibold">{title}</p>

	{#if showWorkingDirectory}
		<button
			class="pointer-events-auto max-w-full truncate text-[13px] leading-4 text-muted-foreground"
			onclick={onWorkingDirectoryClick}
			title={directory ?? undefined}
			type="button"
		>
			{directoryLabel}
		</button>
	{/if}
</div>
