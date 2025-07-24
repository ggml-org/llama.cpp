<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import { Server, Cpu, Eye, Mic } from '@lucide/svelte';
	import { serverStore } from '$lib/stores/server.svelte';

	// Reactive server information
	const props = $derived(serverStore.serverProps);
	const model = $derived(serverStore.modelName);
	const modalities = $derived(serverStore.supportedModalities);
</script>

{#if props}
	<Card class="border-border/20 mb-6 shadow-sm">
		<div class="p-4">
			<div class="mb-3 flex items-center gap-2">
				<Server class="text-muted-foreground h-4 w-4" />
				<h3 class="text-sm font-semibold">Server Info</h3>
			</div>

			<div class="text-muted-foreground space-y-2 text-sm">
				<!-- Model Information -->
				{#if model}
					<div class="flex items-center gap-2">
						<Cpu class="h-3 w-3" />
						<span class="font-medium">Model:</span>
						<span class="font-mono text-xs">{model}</span>
					</div>
				{/if}

				<!-- Build Information -->
				{#if props.build_info}
					<div class="flex items-center gap-2">
						<Server class="h-3 w-3" />
						<span class="font-medium">Build:</span>
						<span class="font-mono text-xs">{props.build_info}</span>
					</div>
				{/if}

				<!-- Context Length -->
				{#if props.n_ctx}
					<div class="flex items-center gap-2">
						<span class="font-medium">Context:</span>
						<span class="font-mono text-xs">{props.n_ctx.toLocaleString()} tokens</span>
					</div>
				{/if}

				<!-- Supported Modalities -->
				{#if modalities.length > 0}
					<div class="flex items-center gap-2">
						<span class="font-medium">Modalities:</span>
						<div class="flex gap-1">
							{#each modalities as modality}
								<Badge variant="secondary" class="h-5 px-2 text-xs">
									{#if modality === 'vision'}
										<Eye class="mr-1 h-3 w-3" />
									{:else if modality === 'audio'}
										<Mic class="mr-1 h-3 w-3" />
									{/if}
									{modality}
								</Badge>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		</div>
	</Card>
{/if}
