<script lang="ts">
	import { BadgeInfo } from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { copyToClipboard } from '$lib/utils';
	import type { Component } from 'svelte';

	interface Props {
		class?: string;
		icon: Component;
		value: string | number;
		tooltip?: string;
	}

	let { class: className = '', icon: Icon, value, tooltip }: Props = $props();

	function handleClick() {
		void copyToClipboard(String(value));
	}
</script>

{#if tooltip}
	<Tooltip.Root>
		<Tooltip.Trigger>
			<BadgeInfo class={className} onclick={handleClick}>
				{#snippet icon()}
					<Icon class="h-3 w-3" />
				{/snippet}

				{value}
			</BadgeInfo>
		</Tooltip.Trigger>
		<Tooltip.Content>
			<p>{tooltip}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{:else}
	<BadgeInfo class={className} onclick={handleClick}>
		{#snippet icon()}
			<Icon class="h-3 w-3" />
		{/snippet}

		{value}
	</BadgeInfo>
{/if}
