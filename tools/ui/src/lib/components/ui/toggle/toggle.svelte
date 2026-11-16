<script lang="ts" module>
	import { tv, type VariantProps } from 'tailwind-variants';

	export const toggleVariants = tv({
		base: "focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive gap-1 rounded-md text-sm font-medium outline-none transition-all focus-visible:ring-[3px] inline-flex items-center justify-center whitespace-nowrap disabled:pointer-events-none disabled:opacity-50 [&_svg:not([class*='size-'])]:size-4 [&_svg]:pointer-events-none [&_svg]:shrink-0 group/toggle",
		defaultVariants: {
			size: 'default',
			variant: 'default'
		},
		variants: {
			size: {
				default:
					'h-8 min-w-8 px-2.5 has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2',
				lg: 'h-9 min-w-9 px-2.5 has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2',
				sm: "h-7 min-w-7 rounded-md px-2.5 text-[0.8rem] has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 [&_svg:not([class*='size-'])]:size-3.5"
			},
			variant: {
				default:
					'bg-transparent hover:bg-muted-foreground/10 hover:text-accent-foreground backdrop-blur-sm data-[state=on]:bg-accent data-[state=on]:text-accent-foreground',
				outline:
					'border bg-background shadow-sm hover:bg-muted-foreground/10 hover:text-accent-foreground backdrop-blur-sm data-[state=on]:bg-accent data-[state=on]:text-accent-foreground dark:border-input'
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
