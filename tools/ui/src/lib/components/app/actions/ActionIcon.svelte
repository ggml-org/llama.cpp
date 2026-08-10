<script lang="ts">
	import { Button, type ButtonSize, type ButtonVariant } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { TooltipSide } from '$lib/enums';
	import type { Component } from 'svelte';

	interface Props {
		ariaLabel?: string;
		class?: string;
		disabled?: boolean;
		href?: string;
		icon: Component;
		iconSize?: string;
		onclick?: (e?: MouseEvent) => void;
		size?: ButtonSize;
		stopPropagationOnClick?: boolean;
		tooltip?: string;
		variant?: ButtonVariant;
		tooltipSide?: TooltipSide;
	}

	let {
		ariaLabel,
		class: className = '',
		disabled = false,
		href = '',
		icon,
		iconSize = 'h-3 w-3',
		onclick,
		size = 'sm',
		stopPropagationOnClick = false,
		tooltip,
		tooltipSide = TooltipSide.TOP,
		variant = 'ghost'
	}: Props = $props();

	let innerWidth = $state(0);
	const showTooltip = $derived(!!tooltip && innerWidth > 768);
</script>

{#snippet button(props = {})}
	<Button
		{...props}
		{href}
		{variant}
		{size}
		{disabled}
		onclick={(e: MouseEvent) => {
			if (stopPropagationOnClick) e.stopPropagation();

			onclick?.(e);
		}}
		class="h-6 w-6 p-0 {className} flex hover:bg-transparent data-[state=open]:bg-transparent!"
		aria-label={ariaLabel || tooltip}
	>
		{#if icon}
			{@const IconComponent = icon}

			<IconComponent class={iconSize} />
		{/if}
	</Button>
{/snippet}

{#if showTooltip}
	<Tooltip.Root>
		<Tooltip.Trigger>
			<!-- prevent another nested button element -->
			{#snippet child({ props })}
				{#if disabled}
					<!-- disabled buttons have pointer-events:none; wrap in a span so the tooltip hover surface stays alive -->
					<span {...props}>
						{@render button({})}
					</span>
				{:else}
					{@render button(props)}
				{/if}
			{/snippet}
		</Tooltip.Trigger>

		<Tooltip.Content side={tooltipSide}>
			<p>{tooltip}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{:else}
	{@render button({ href })}
{/if}

<svelte:window bind:innerWidth />
