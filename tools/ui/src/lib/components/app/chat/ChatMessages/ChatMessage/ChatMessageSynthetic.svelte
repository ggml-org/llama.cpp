<script lang="ts">
	import { parseCwdMessage } from '$lib/utils';
	import type { DatabaseMessage } from '$lib/types';
	import ChatMessageCwdChange from './ChatMessageCwdChange.svelte';

	interface Props {
		class?: string;
		message: DatabaseMessage;
	}

	let { class: className = '', message }: Props = $props();

	// Synthetic messages render a dedicated UI, never a user bubble. The only
	// kind today is the working-directory change; parse the content so the
	// row reuses the exact synthetic text (and future kinds slot in here).
	let isCwdChange = $derived(parseCwdMessage(message.content) !== null);
</script>

{#if isCwdChange}
	<ChatMessageCwdChange {message} class={className} />
{:else}
	<span class="text-muted-foreground block text-sm {className}">{message.content}</span>
{/if}
