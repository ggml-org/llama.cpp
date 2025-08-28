<script lang="ts">
	import type { ComponentProps } from 'svelte';
	import { Input } from '$lib/components/ui/input/index.js';
	import { cn } from '$lib/components/ui/utils.js';

	let {
		ref = $bindable(null),
		value = $bindable(''),
		class: className,
		...restProps
	}: ComponentProps<typeof Input> = $props();
</script>

<Input
	bind:ref
	bind:value
	data-slot="sidebar-input"
	data-sidebar="input"
	class={cn('h-8 w-full bg-background shadow-none', className)}
	{...restProps}
/>
