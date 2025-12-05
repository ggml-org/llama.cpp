<script lang="ts">
	import { Package } from '@lucide/svelte';
	import { BadgeInfo, CopyToClipboardIcon } from '$lib/components/app';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { serverStore } from '$lib/stores/server.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		class?: string;
		model?: string;
		onclick?: () => void;
		showCopyIcon?: boolean;
		showTooltip?: boolean;
	}

	let {
		class: className = '',
		model: modelProp,
		onclick,
		showCopyIcon = false,
		showTooltip = false
	}: Props = $props();

	let model = $derived(modelProp || modelsStore.singleModelName);
	let isModelMode = $derived(serverStore.isModelMode);
</script>

{#snippet badgeContent()}
	<BadgeInfo class={className} {onclick}>
		{#snippet icon()}
			<Package class="h-3 w-3" />
		{/snippet}

		{model}

		{#if showCopyIcon}
			<CopyToClipboardIcon text={model || ''} ariaLabel="Copy model name" />
		{/if}
	</BadgeInfo>
{/snippet}

{#if model && isModelMode}
	{#if showTooltip}
		<Tooltip.Root>
			<Tooltip.Trigger>
				{@render badgeContent()}
			</Tooltip.Trigger>

			<Tooltip.Content>
				{onclick ? 'Click for model details' : model}
			</Tooltip.Content>
		</Tooltip.Root>
	{:else}
		{@render badgeContent()}
	{/if}
{/if}
