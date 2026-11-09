<script lang="ts">
	import ChatMessageCwdChangeRow from './ChatMessageCwdChangeRow.svelte';
	import { FolderX } from '@lucide/svelte';
	import type { DatabaseMessage } from '$lib/types';
	import { parseCwdMessage } from '$lib/utils';

	interface Props {
		class?: string;
		message: DatabaseMessage;
	}

	let { class: className = '', message }: Props = $props();

	// Parse the synthetic message content in the UI so the row reuses the
	// exact same text the model saw, including any guidance suffix.
	let info = $derived(parseCwdMessage(message.content));
</script>

{#if info}
	{#if info.path === null}
		<div class="text-muted-foreground flex items-center gap-2 py-1.5 {className}">
			<FolderX class="text-muted-foreground/60 size-icon-sm shrink-0" />

			<span class="text-foreground/80 text-sm font-medium">Working directory cleared</span>
		</div>
	{:else}
		<ChatMessageCwdChangeRow class={className} display={info.display} path={info.path} />
	{/if}
{/if}
