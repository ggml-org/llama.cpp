<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { cn, type WithElementRef } from '$lib/components/ui/utils';

	let {
		ref = $bindable(null),
		class: className,
		children,
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLDivElement>> = $props();
</script>

<div
	bind:this={ref}
	data-slot="card"
	class={cn(
		'bg-card text-card-foreground flex flex-col gap-6 rounded-xl border py-6 shadow-sm',
		className
	)}
	{...restProps}
>
	{@render children?.()}
</div>
