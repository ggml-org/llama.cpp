<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import { Server, Eye, Mic } from '@lucide/svelte';
	import { serverStore } from '$lib/stores/server.svelte';

	const props = $derived(serverStore.serverProps);
	const model = $derived(serverStore.modelName);
	const modalities = $derived(serverStore.supportedModalities);
</script>

{#if props}
	<div class="text-muted-foreground flex items-center justify-center gap-3 text-sm">
		{#if model}
			<Badge variant="outline" class="text-xs">
				<Server class="mr-1 h-3 w-3" />
				<span class="block max-w-[50vw] truncate">{model}</span>
			</Badge>
		{/if}

		{#if props.default_generation_settings.n_ctx}
			<Badge variant="secondary" class="text-xs">
				ctx: {props.default_generation_settings.n_ctx.toLocaleString()}
			</Badge>
		{/if}

		{#if modalities.length > 0}
			{#each modalities as modality}
				<Badge variant="secondary" class="text-xs">
					{#if modality === 'vision'}
						<Eye class="mr-1 h-3 w-3" />
					{:else if modality === 'audio'}
						<Mic class="mr-1 h-3 w-3" />
					{/if}
					{modality}
				</Badge>
			{/each}
		{/if}
	</div>
{/if}
