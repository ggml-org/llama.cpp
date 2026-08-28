<script lang="ts" module>
	import { tv, type VariantProps } from 'tailwind-variants';

	export const badgeVariants = tv({
		base: 'focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive inline-flex w-fit shrink-0 items-center justify-center gap-1 overflow-hidden whitespace-nowrap rounded-md border px-2 py-0.5 text-xs font-medium transition-[color,box-shadow] focus-visible:ring-[3px] [&>svg]:pointer-events-none [&>svg]:size-3',
		defaultVariants: {
			variant: 'default'
		},
		variants: {
			variant: {
				default: 'bg-primary text-primary-foreground [a&]:hover:bg-primary/90 border-transparent',
				destructive:
					'bg-destructive [a&]:hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/70 border-transparent text-white',
				outline: 'text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground',
				secondary:
					'bg-secondary text-secondary-foreground [a&]:hover:bg-secondary/90 border-transparent',
				tertiary:
					'bg-foreground/15 dark:bg-foreground/10 text-foreground [a&]:hover:bg-foreground/25 border-transparent'
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
	class={cn(badgeVariants({ variant }), className, 'backdrop-blur-sm')}
	data-slot="badge"
	{href}
	{...restProps}
>
	{@render children?.()}
</svelte:element>
