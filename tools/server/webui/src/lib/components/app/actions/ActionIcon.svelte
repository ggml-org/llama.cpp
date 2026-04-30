<script lang="ts">
	import { Button, type ButtonVariant, type ButtonSize } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { Component } from 'svelte';
	import { TooltipSide } from '$lib/enums';

	interface Props {
		ariaLabel?: string;
		class?: string;
		disabled?: boolean;
		icon: Component;
		iconSize?: string;
		onclick: (e?: MouseEvent) => void;
		size?: ButtonSize;
		stopPropagationOnClick?: boolean;
		tooltip: string;
		variant?: ButtonVariant;
		tooltipSide?: TooltipSide;
	}

	let {
		icon,
		tooltip,
		variant = 'ghost',
		size = 'sm',
		class: className = '',
		disabled = false,
		iconSize = 'h-3 w-3',
		tooltipSide = TooltipSide.TOP,
		stopPropagationOnClick = false,
		onclick,
		ariaLabel
	}: Props = $props();
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		<Button
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
	</Tooltip.Trigger>

	<Tooltip.Content side={tooltipSide}>
		<p>{tooltip}</p>
	</Tooltip.Content>
</Tooltip.Root>
