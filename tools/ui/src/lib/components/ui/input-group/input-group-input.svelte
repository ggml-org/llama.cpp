<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import type { ComponentProps } from 'svelte';
	import { Input } from '$lib/components/ui/input/index.js';

	let {
		ref = $bindable(null),
		value = $bindable(),
		class: className,
		...props
	}: ComponentProps<typeof Input> = $props();
</script>

<Input
	bind:ref
	data-slot="input-group-control"
	class={cn(
		'rounded-none border-0 bg-transparent shadow-none ring-0 focus-visible:ring-0 disabled:bg-transparent aria-invalid:ring-0 dark:bg-transparent dark:disabled:bg-transparent flex-1',
		className
	)}
	bind:value
	{...props}
/>
