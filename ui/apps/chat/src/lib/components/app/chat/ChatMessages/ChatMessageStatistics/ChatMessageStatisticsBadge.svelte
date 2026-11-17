<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { BadgeInfo } from '#lib/components/app/index.ts';
	import { copyToClipboard } from '#lib/utils/index.ts';
	import type { Component } from 'svelte';

	interface Props {
		class?: string;
		icon: Component;
		value: string | number;
		tooltipLabel?: string;
	}

	let { class: className = '', icon: IconComponent, tooltipLabel, value }: Props = $props();

	function handleClick() {
		void copyToClipboard(String(value));
	}
</script>

{#if tooltipLabel}
	<Tooltip.Root>
		<Tooltip.Trigger>
			<!-- prevent another nested button element -->
			{#snippet child({ props })}
				<BadgeInfo {...props} class={className} onclick={handleClick}>
					{#snippet icon()}
						<IconComponent class="size-icon-xs" />
					{/snippet}

					{value}
				</BadgeInfo>
			{/snippet}
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>{tooltipLabel}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{:else}
	<BadgeInfo class={className} onclick={handleClick}>
		{#snippet icon()}
			<IconComponent class="size-icon-xs" />
		{/snippet}

		{value}
	</BadgeInfo>
{/if}
