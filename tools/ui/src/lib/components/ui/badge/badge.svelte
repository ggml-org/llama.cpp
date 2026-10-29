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
				default:
					'bg-(--badge-primary-background) text-(--badge-primary-text) [a&]:hover:bg-(--badge-primary-background-hover) border-transparent',
				destructive:
					'bg-(--badge-danger-background) [a&]:hover:bg-(--badge-danger-background-hover) focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 border-transparent text-(--badge-danger-text)',
				outline:
					'text-(--badge-text) border-(--badge-outline-border) [a&]:hover:bg-(--badge-outline-background-hover)',
				secondary:
					'bg-(--badge-secondary-background) text-(--badge-text) [a&]:hover:bg-(--badge-secondary-background-hover) border-transparent'
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
	class={cn(badgeVariants({ variant }), className, 'matte-chip backdrop-blur-sm')}
	data-slot="badge"
	{href}
	{...restProps}
>
	{@render children?.()}
</svelte:element>
