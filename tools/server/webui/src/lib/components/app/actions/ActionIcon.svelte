<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { Component } from 'svelte';

	interface Props {
		icon: Component;
		tooltip: string;
		variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
		size?: 'default' | 'sm' | 'lg' | 'icon';
		iconSize?: string;
		class?: string;
		disabled?: boolean;
		onclick: () => void;
		'aria-label'?: string;
	}

	let {
		icon,
		tooltip,
		variant = 'ghost',
		size = 'sm',
		class: className = '',
		disabled = false,
		iconSize = 'h-3 w-3',
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
			class="h-6 w-6 p-0 {className} flex"
			aria-label={ariaLabel || tooltip}
		>
			{@const IconComponent = icon}

			<IconComponent class={iconSize} />
		</Button>
	</Tooltip.Trigger>

	<Tooltip.Content>
		<p>{tooltip}</p>
	</Tooltip.Content>
</Tooltip.Root>
