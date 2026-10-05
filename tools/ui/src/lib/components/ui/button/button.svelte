<script lang="ts" module>
	import { cn, type WithElementRef } from '$lib/components/ui/utils';
	import type { HTMLAnchorAttributes, HTMLButtonAttributes } from 'svelte/elements';
	import { tv, type VariantProps } from 'tailwind-variants';

	export const buttonVariants = tv({
		base: "focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium outline-none transition-all focus-visible:ring-[3px] disabled:pointer-events-none disabled:opacity-50 aria-disabled:pointer-events-none aria-disabled:opacity-50 [&_svg:not([class*='size-'])]:size-4 [&_svg]:pointer-events-none [&_svg]:shrink-0",
		defaultVariants: {
			size: 'default',
			variant: 'default'
		},
		variants: {
			size: {
				default: 'h-9 px-4 py-2 has-[>svg]:px-3',
				icon: 'size-9',
				'icon-lg': 'size-10',
				'icon-sm': 'size-5 rounded-sm',
				lg: 'h-10 rounded-lg px-6 has-[>svg]:px-4',
				sm: 'h-8 gap-1.5 rounded-md px-3 has-[>svg]:px-2.5'
			},
			variant: {
				default: 'bg-primary text-primary-foreground shadow-sm hover:bg-primary/90',
				destructive:
					'bg-destructive shadow-sm hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60 text-white!',
				ghost: 'hover:text-accent-foreground hover:bg-muted-foreground/10 backdrop-blur-sm',
				link: 'text-primary underline-offset-4 hover:underline',
				outline:
					'shadow-sm hover:text-accent-foreground hover:bg-muted-foreground/10 backdrop-blur-sm dark:border-input border',
				secondary:
					'bg-background dark:bg-muted-foreground/15 dark:text-secondary-foreground shadow-sm text-foreground hover:bg-muted-foreground/20 dark:hover:bg-muted-foreground/25'
			}
		}
	});

	export type ButtonVariant = VariantProps<typeof buttonVariants>['variant'];
	export type ButtonSize = VariantProps<typeof buttonVariants>['size'];

	export type ButtonProps = WithElementRef<HTMLButtonAttributes> &
		WithElementRef<HTMLAnchorAttributes> & {
			variant?: ButtonVariant;
			size?: ButtonSize;
		};
</script>

<script lang="ts">
	let {
		children,
		class: className,
		disabled,
		href = undefined,
		ref = $bindable(null),
		size = 'default',
		type = 'button',
		variant = 'default',
		...restProps
	}: ButtonProps = $props();
</script>

{#if href}
	<a
		bind:this={ref}
		aria-disabled={disabled}
		class={cn(buttonVariants({ size, variant }), className)}
		data-slot="button"
		href={disabled ? undefined : href}
		role={disabled ? 'link' : undefined}
		tabindex={disabled ? -1 : undefined}
		{...restProps}
	>
		{@render children?.()}
	</a>
{:else}
	<button
		bind:this={ref}
		class={cn(buttonVariants({ size, variant }), className)}
		data-slot="button"
		{disabled}
		{type}
		{...restProps}
	>
		{@render children?.()}
	</button>
{/if}

<style>
	a,
	button {
		cursor: pointer;
	}
</style>
