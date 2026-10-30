<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	interface Props extends HTMLAttributes<HTMLDivElement> {
		children: Snippet;
	}

	let { children, class: className, ...restProps }: Props = $props();
</script>

<!--
	joined buttons share one shadow, drawn by the group. Bordered buttons share one border
	at each seam, or the separator instead; filled buttons draw a seam in their own text
	colour, so a separator between them is hidden.
-->
<div
	class={cn(
		'flex w-fit items-center rounded-control shadow-(--button-shadow) *:shadow-none *:[--button-shadow:none] *:focus-visible:z-10',
		'[&>*:not(:first-child)]:rounded-l-none [&>*:not(:first-child)]:border-l-0 [&>*:not(:last-child)]:rounded-r-none [&>*:has(+[data-slot=button-group-separator])]:border-r-0',
		'[&>[data-variant=default]:not(:first-child)]:border-l [&>[data-variant=default]]:border-l-(--button-primary-seam) [&>[data-variant=destructive]:not(:first-child)]:border-l [&>[data-variant=destructive]]:border-l-(--button-danger-seam)',
		'[&>[data-slot=button-group-separator]:has(+[data-variant=default],+[data-variant=destructive])]:hidden',
		className
	)}
	data-slot="button-group"
	{...restProps}
>
	{@render children()}
</div>
