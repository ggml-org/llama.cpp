<script lang="ts">
	import { Folder, FolderX } from '@lucide/svelte';
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
	<div class="text-muted-foreground flex items-center gap-2 py-1.5 {className}">
		{#if info.path === null}
			<FolderX class="text-muted-foreground/60 h-3.5 w-3.5 shrink-0" />
			<span class="text-foreground/80 text-sm font-medium">Working directory cleared</span>
		{:else}
			<Folder class="text-muted-foreground/60 h-3.5 w-3.5 shrink-0" />
			<span class="text-foreground/80 text-sm font-medium">Set working directory to&nbsp;</span>
			<span class="font-mono text-foreground/90 text-sm break-all" title={info.path}>
				{info.display}
			</span>
		{/if}
	</div>
{/if}
