<script lang="ts">
	import { cn, type WithElementRef } from '../utils';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		children,
		class: className,
		ref = $bindable(null),
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLParagraphElement>> = $props();
</script>

<p
	bind:this={ref}
	class={cn('text-body-sm text-muted-foreground', className)}
	data-slot="card-description"
	{...restProps}
>
	{@render children?.()}
</p>
