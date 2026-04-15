<script lang="ts">
	import { Button, type ButtonVariant, type ButtonSize } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { Component } from 'svelte';
	import { TooltipSide } from '$lib/enums';

	interface Props {
		icon: Component;
		tooltip: string;
		variant?: ButtonVariant;
		size?: ButtonSize;
		iconSize?: string;
		class?: string;
		disabled?: boolean;
		onclick: (e?: MouseEvent) => void;
		'aria-label'?: string;
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
		onclick,
		'aria-label': ariaLabel
	}: Props = $props();
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		<Button
			{variant}
			{size}
			{disabled}
			{onclick}
			class="h-6 w-6 p-0 {className} flex hover:bg-transparent data-[state=open]:bg-transparent!"
			aria-label={ariaLabel || tooltip}
		>
			{@const IconComponent = icon}

			<IconComponent class={iconSize} />
		</Button>
	</Tooltip.Trigger>

	<Tooltip.Content side={tooltipSide}>
		<p>{tooltip}</p>
	</Tooltip.Content>
</Tooltip.Root>
