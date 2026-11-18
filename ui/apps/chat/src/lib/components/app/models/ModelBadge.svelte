<script lang="ts">
	import ModelId from './ModelId.svelte';
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { ActionIconCopyToClipboard, BadgeInfo } from '#lib/components/app/index.ts';
	import { MODEL_ICON } from '#lib/constants/index.ts';
	import { modelsStore, serverStore } from '#lib/stores/index.ts';

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
	let shouldShow = $derived(model && (modelProp !== undefined || isModelMode));
</script>

{#snippet badgeContent(triggerProps?: Record<string, unknown>)}
	<BadgeInfo {...triggerProps ?? {}} class={className} {onclick}>
		{#snippet icon()}
			<MODEL_ICON class="size-icon-xs" />
		{/snippet}

		{#if model}
			<ModelId modelId={model} />
		{/if}

		{#if showCopyIcon}
			<ActionIconCopyToClipboard ariaLabel="Copy model name" text={model || ''} />
		{/if}
	</BadgeInfo>
{/snippet}

{#if shouldShow}
	{#if showTooltip}
		<Tooltip.Root>
			<Tooltip.Trigger>
				<!-- prevent another nested button element -->
				{#snippet child({ props })}
					{@render badgeContent(props)}
				{/snippet}
			</Tooltip.Trigger>

			<Tooltip.Content>
				{onclick ? 'Click for model details' : model}
			</Tooltip.Content>
		</Tooltip.Root>
	{:else}
		{@render badgeContent()}
	{/if}
{/if}
