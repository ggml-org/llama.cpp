<script lang="ts">
	import { chatStore } from '$lib/stores/chat.svelte';
	import { StreamConnectionState } from '$lib/enums';
	import { Loader2 } from '@lucide/svelte';

	let state = $derived(chatStore.streamConnectionState);
</script>

{#if state === StreamConnectionState.RESUMING}
	<div
		class="pointer-events-auto mx-auto mt-2 mb-2 flex max-w-[48rem] items-center gap-2 rounded-md border border-blue-400/40 bg-blue-50/60 px-3 py-1.5 text-sm text-blue-700 dark:bg-blue-950/40 dark:text-blue-200"
		role="status"
		aria-live="polite"
	>
		<Loader2 class="h-3.5 w-3.5 animate-spin" />
		<span>Reconnecting to the stream...</span>
	</div>
{/if}
