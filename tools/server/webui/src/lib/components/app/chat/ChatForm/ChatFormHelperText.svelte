<script lang="ts">
	import { browser } from '$app/environment';
	import { config } from '$lib/stores/settings.svelte';

	interface Props {
		class?: string;
		show?: boolean;
	}

	let { class: className = '', show = true }: Props = $props();

	let sendOnEnter = $derived(config().sendOnEnter !== false);
	let modKey = browser && /Mac|iPhone|iPad|iPod/.test(navigator.platform) ? 'Cmd' : 'Ctrl';
</script>

{#if show}
	<div class="mt-6 items-center justify-center {className} hidden md:flex">
		{#if sendOnEnter}
			<p class="text-xs text-muted-foreground">
				Press <kbd class="rounded bg-muted px-1 py-0.5 font-mono text-xs">Enter</kbd> to send,
				<kbd class="rounded bg-muted px-1 py-0.5 font-mono text-xs">Shift + Enter</kbd> for new line
			</p>
		{:else}
			<p class="text-xs text-muted-foreground">
				Press <kbd class="rounded bg-muted px-1 py-0.5 font-mono text-xs">{modKey} + Enter</kbd> to
				send,
				<kbd class="rounded bg-muted px-1 py-0.5 font-mono text-xs">Enter</kbd> for new line
			</p>
		{/if}
	</div>
{/if}
