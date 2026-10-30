<script lang="ts" module>
	import { cn, type WithElementRef } from '$lib/components/ui/utils';
	import { tv } from '$lib/components/ui/utils';
	import type { HTMLAnchorAttributes, HTMLButtonAttributes } from 'svelte/elements';
	import type { VariantProps } from 'tailwind-variants';

	export const buttonVariants = tv({
		base: "focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-control text-label font-medium outline-none transition-all focus-visible:ring-[3px] disabled:pointer-events-none disabled:opacity-50 aria-disabled:pointer-events-none aria-disabled:opacity-50 [&_svg:not([class*='size-'])]:size-icon-md [&_svg]:pointer-events-none [&_svg]:shrink-0",
		defaultVariants: {
			size: 'default',
			variant: 'default'
		},
		variants: {
			size: {
				default: 'h-control-md px-4 py-2 has-[>svg]:px-3',
				icon: 'size-(--height-control-md)',
				'icon-lg': 'size-(--height-control-lg)',
				'icon-sm': 'size-5 rounded-sm',
				lg: 'h-control-lg rounded-panel px-6 has-[>svg]:px-4',
				sm: 'h-control-sm gap-1.5 rounded-control px-3 has-[>svg]:px-2.5'
			},
			variant: {
				default:
					'bg-(--button-primary-background) text-(--button-primary-text) shadow-sm hover:bg-(--button-primary-background-hover)',
				destructive:
					'bg-(--button-danger-background) shadow-sm hover:bg-(--button-danger-background-hover) focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 text-(--button-danger-text)!',
				ghost:
					'hover:text-(--button-text) hover:bg-(--button-ghost-background-hover) backdrop-blur-sm',
				outline:
					'matte-key bg-(--button-outline-background) hover:text-(--button-text) hover:bg-(--button-ghost-background-hover) border border-(--button-outline-border)',
				secondary:
					'matte-key bg-(--button-secondary-background) border border-(--button-secondary-border) text-(--button-text) hover:bg-(--button-secondary-background-hover)',
				tertiary:
					'matte-key bg-(--button-tertiary-background) border border-(--button-tertiary-border) text-(--button-text) hover:bg-(--button-tertiary-background-hover)'
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
		data-variant={variant}
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
		data-variant={variant}
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
