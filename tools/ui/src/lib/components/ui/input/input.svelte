<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils';
	import type { HTMLInputAttributes, HTMLInputTypeAttribute } from 'svelte/elements';

	type InputType = Exclude<HTMLInputTypeAttribute, 'file'>;

	type Props = WithElementRef<
		Omit<HTMLInputAttributes, 'type'> &
			({ type: 'file'; files?: FileList } | { type?: InputType; files?: undefined })
	>;

	let {
		class: className,
		files = $bindable(),
		ref = $bindable(null),
		type,
		value = $bindable(),
		...restProps
	}: Props = $props();
</script>

{#if type === 'file'}
	<input
		bind:files
		bind:this={ref}
		bind:value
		class={cn(
			'flex h-control-md w-full min-w-0 rounded-control border border-(--input-border) bg-transparent px-3 pt-1.5 text-label font-medium matte-well ring-offset-background transition-[color,box-shadow] outline-none selection:bg-primary selection:text-primary-foreground placeholder:text-(--input-placeholder) disabled:cursor-not-allowed disabled:opacity-50 dark:bg-(--input-background)',
			'focus-visible:border-(--input-border-focus) focus-visible:ring-[3px] focus-visible:ring-ring/50',
			'aria-invalid:border-(--input-border-invalid) aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40',
			className
		)}
		data-slot="input"
		type="file"
		{...restProps}
	/>
{:else}
	<input
		bind:this={ref}
		bind:value
		class={cn(
			'flex h-control-md w-full min-w-0 rounded-control border border-(--input-border) bg-(--input-background) px-3 py-1 text-body matte-well ring-offset-background transition-[color,box-shadow] outline-none selection:bg-primary selection:text-primary-foreground placeholder:text-(--input-placeholder) disabled:cursor-not-allowed disabled:opacity-50 md:text-body-sm',
			'focus-visible:border-(--input-border-focus) focus-visible:ring-[3px] focus-visible:ring-ring/50',
			'aria-invalid:border-(--input-border-invalid) aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40',
			className
		)}
		data-slot="input"
		style="backdrop-filter: blur(0.5rem);"
		{type}
		{...restProps}
	/>
{/if}
