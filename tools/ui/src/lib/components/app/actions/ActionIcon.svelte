<script lang="ts">
	import { Button, type ButtonSize, type ButtonVariant } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';
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
		tooltipAsTitle?: boolean;
		tooltipSide?: TooltipSide;
		variant?: ButtonVariant;
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
		tooltipAsTitle = false,
		tooltipSide = TooltipSide.TOP,
		variant = 'ghost'
	}: Props = $props();

	let innerWidth = $state(0);
	const showTooltip = $derived(!!tooltip && !tooltipAsTitle && innerWidth > 768);
</script>

{#snippet button(props = {})}
	<Button
		{...props}
		aria-label={ariaLabel || tooltip}
		class={cn(
			'flex h-6 w-6 p-0 hover:bg-transparent data-[state=open]:bg-transparent!',
			disabled && 'disabled:opacity-40',
			className
		)}
		{disabled}
		{href}
		onclick={(e: MouseEvent) => {
			if (stopPropagationOnClick) e.stopPropagation();

			onclick?.(e);
		}}
		{size}
		title={tooltipAsTitle ? tooltip : undefined}
		{variant}
	>
		{#if icon}
			{@const IconComponent = icon}

			<IconComponent class={iconSize} />
		{/if}
	</Button>
{/snippet}

{#if showTooltip}
	<Tooltip.Root>
		<Tooltip.Trigger class={disabled ? 'cursor-not-allowed' : undefined}>
			<!-- prevent another nested button element -->
			{#snippet child({ props })}
				{#if disabled}
					<!-- disabled buttons have pointer-events:none; wrap in a span so the tooltip hover surface stays alive -->
					<span
						{...props}
						class={cn(props.class as string | undefined, 'inline-flex cursor-not-allowed')}
					>
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
{:else if disabled}
	<span class="inline-flex cursor-not-allowed" title={tooltipAsTitle ? tooltip : undefined}>
		{@render button({ href })}
	</span>
{:else}
	{@render button({ href })}
{/if}

<svelte:window bind:innerWidth />
