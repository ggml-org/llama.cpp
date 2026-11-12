<script lang="ts" module>
	import { tv } from '$lib/components/ui/utils';
	import type { VariantProps } from 'tailwind-variants';

	export const badgeVariants = tv({
		base: 'focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive inline-flex w-fit shrink-0 items-center justify-center gap-1 overflow-hidden whitespace-nowrap rounded-control border px-2 py-0.5 text-label-sm font-medium transition-[color,box-shadow] focus-visible:ring-[3px] [&>svg]:pointer-events-none [&>svg]:size-icon-xs',
		defaultVariants: {
			variant: 'default'
		},
		variants: {
			variant: {
				// soft status badges: a pill in the status colour, a dot before the label, on its own soft surface and line
				danger:
					"matte-sheen bg-surface-danger text-fg-danger border-line-danger-soft rounded-full gap-1.5 before:size-1.5 before:shrink-0 before:rounded-full before:bg-fill-danger before:content-['']",
				default:
					'matte-sheen-solid bg-(--badge-primary-background) text-(--badge-primary-text) [a&]:hover:bg-(--badge-primary-background-hover) border-transparent',
				// solid, for what needs attention now
				destructive:
					'matte-sheen-solid bg-(--badge-danger-background) [a&]:hover:bg-(--badge-danger-background-hover) focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 border-transparent text-(--badge-danger-text)',
				info: "matte-sheen bg-surface-info text-fg-info border-line-info-soft rounded-full gap-1.5 before:size-1.5 before:shrink-0 before:rounded-full before:bg-fill-info before:content-['']",
				outline:
					'text-(--badge-text) border-(--badge-outline-border) [a&]:hover:bg-(--badge-outline-background-hover)',
				secondary:
					'matte-sheen bg-(--badge-secondary-background) text-(--badge-text) [a&]:hover:bg-(--badge-secondary-background-hover) border-transparent',
				success:
					"matte-sheen bg-surface-success text-fg-success border-line-success-soft rounded-full gap-1.5 before:size-1.5 before:shrink-0 before:rounded-full before:bg-fill-success before:content-['']",
				warning:
					"matte-sheen bg-surface-warning text-fg-warning border-line-warning-soft rounded-full gap-1.5 before:size-1.5 before:shrink-0 before:rounded-full before:bg-fill-warning before:content-['']"
			}
		}
	});

	export type BadgeVariant = VariantProps<typeof badgeVariants>['variant'];
</script>

<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils';
	import type { HTMLAnchorAttributes } from 'svelte/elements';

	let {
		children,
		class: className,
		href,
		ref = $bindable(null),
		variant = 'default',
		...restProps
	}: WithElementRef<HTMLAnchorAttributes> & {
		variant?: BadgeVariant;
	} = $props();
</script>

<svelte:element
	this={href ? 'a' : 'span'}
	bind:this={ref}
	class={cn(badgeVariants({ variant }), className)}
	data-slot="badge"
	{href}
	{...restProps}
>
	{@render children?.()}
</svelte:element>
