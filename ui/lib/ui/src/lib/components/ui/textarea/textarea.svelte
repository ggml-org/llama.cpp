<script lang="ts">
	import { cn, type WithElementRef, type WithoutChildren } from '../utils';
	import type { HTMLTextareaAttributes } from 'svelte/elements';

	let {
		class: className,
		ref = $bindable(null),
		value = $bindable(),
		...restProps
	}: WithoutChildren<WithElementRef<HTMLTextareaAttributes>> = $props();
</script>

<textarea
	bind:this={ref}
	bind:value
	class={cn(
		'flex field-sizing-content min-h-16 w-full rounded-md border border-(--line-field) bg-(--field-background) px-3 py-2 text-body matte-well transition-[color,box-shadow] outline-none placeholder:text-(--fg-muted) focus-visible:border-(--line-focus) focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-(--line-danger) aria-invalid:ring-destructive/20 md:text-body-sm dark:aria-invalid:ring-destructive/40',
		className
	)}
	data-slot="textarea"
	{...restProps}
></textarea>
