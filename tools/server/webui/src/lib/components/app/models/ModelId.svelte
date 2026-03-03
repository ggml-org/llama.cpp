<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import { ModelsService } from '$lib/services/models.service';
	import { config } from '$lib/stores/settings.svelte';

	interface Props {
		modelId: string;
		showOrgName?: boolean;
		showRaw?: boolean;
		class?: string;
	}

	let {
		modelId,
		showOrgName = false,
		showRaw = undefined,
		class: className = ''
	}: Props = $props();

	let parsed = $derived(ModelsService.parseModelId(modelId));
	let resolvedShowRaw = $derived(showRaw ?? (config().showRawModelNames as boolean) ?? false);
</script>

{#if resolvedShowRaw}
	<span class="min-w-0 truncate font-medium {className}">{modelId}</span>
{:else}
	<span class="flex min-w-0 flex-wrap items-center gap-1 {className}">
		<span class="min-w-0 truncate font-medium">
			{#if showOrgName}{parsed.orgName}/{/if}{parsed.modelName ?? modelId}
		</span>

		{#if parsed.params}
			<Badge variant="tertiary" class="px-1 py-0 font-mono text-[10px]">
				{parsed.params}{parsed.activatedParams ? `-${parsed.activatedParams}` : ''}
			</Badge>
		{/if}

		{#if parsed.quantization}
			<Badge variant="tertiary" class="px-1 py-0 font-mono text-[10px]">
				{parsed.quantization}
			</Badge>
		{/if}
	</span>
{/if}
