<script lang="ts" module>
	import { tv } from '$lib/components/ui/utils';
	import type { VariantProps } from 'tailwind-variants';

	export const toggleVariants = tv({
		base: "focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive gap-2 rounded-control text-label font-medium outline-none transition-all focus-visible:ring-[3px] inline-flex items-center justify-center whitespace-nowrap disabled:pointer-events-none disabled:opacity-50 aria-disabled:pointer-events-none aria-disabled:opacity-50 [&_svg:not([class*='size-'])]:size-icon-md [&_svg]:pointer-events-none [&_svg]:shrink-0 group/toggle hover:text-(--toggle-text) hover:bg-(--toggle-background-hover) data-[state=on]:bg-(--toggle-background-on) data-[state=on]:hover:bg-(--toggle-background-on) data-[state=on]:text-(--toggle-text)",
		defaultVariants: {
			size: 'default',
			variant: 'default'
		},
		// the same heights, padding and corners as button; icon sizes are square
		variants: {
			size: {
				default: 'h-control-md px-4 has-[>svg]:px-3',
				icon: 'size-(--height-control-md)',
				'icon-lg': 'size-(--height-control-lg) rounded-panel',
				'icon-sm': 'size-(--height-control-sm)',
				lg: 'h-control-lg rounded-panel px-6 has-[>svg]:px-4',
				sm: 'h-control-sm gap-1.5 px-3 has-[>svg]:px-2.5'
			},
			variant: {
				default: 'bg-transparent backdrop-blur-sm',
				outline:
					'matte-key border border-(--button-outline-border) bg-(--button-outline-background)'
			}
		}
	});

	export type ToggleVariant = VariantProps<typeof toggleVariants>['variant'];
	export type ToggleSize = VariantProps<typeof toggleVariants>['size'];
	export type ToggleVariants = VariantProps<typeof toggleVariants>;
</script>

<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { Toggle as TogglePrimitive } from 'bits-ui';

	let {
		class: className,
		pressed = $bindable(false),
		ref = $bindable(null),
		size = 'default',
		variant = 'default',
		...restProps
	}: TogglePrimitive.RootProps & {
		variant?: ToggleVariant;
		size?: ToggleSize;
	} = $props();
</script>

<TogglePrimitive.Root
	bind:pressed
	bind:ref
	class={cn(toggleVariants({ size, variant }), className)}
	data-slot="toggle"
	{...restProps}
/>
