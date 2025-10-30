<script lang="ts">
	import { Server, Eye, Mic } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { serverStore } from '$lib/stores/server.svelte';

	let modalities = $derived(serverStore.supportedModalities);
	let model = $derived(serverStore.modelName);
	let props = $derived(serverStore.serverProps);
</script>

{#if props}
	<div class="flex flex-wrap items-center justify-center gap-4 text-sm text-muted-foreground">
		{#if model}
			<Badge variant="outline" class="text-xs">
				<Server class="mr-1 h-3 w-3" />

				<span class="block max-w-[50vw] truncate">{model}</span>
			</Badge>
		{/if}

		<div class="flex gap-4">
			{#if props.default_generation_settings.n_ctx}
				<Badge variant="secondary" class="text-xs">
					ctx: {props.default_generation_settings.n_ctx.toLocaleString()}
				</Badge>
			{/if}

			{#if modalities.length > 0}
				{#each modalities as modality (modality)}
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
	</div>
{/if}
