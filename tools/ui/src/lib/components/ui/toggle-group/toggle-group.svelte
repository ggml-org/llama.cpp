<script lang="ts" module>
	import { toggleVariants } from '$lib/components/ui/toggle/index.js';
	import { getContext, setContext } from 'svelte';
	import type { VariantProps } from 'tailwind-variants';

	type ToggleVariants = VariantProps<typeof toggleVariants>;

	interface ToggleGroupContext extends ToggleVariants {
		spacing?: number;
		orientation?: 'horizontal' | 'vertical';
	}

	export function setToggleGroupCtx(props: ToggleGroupContext) {
		setContext('toggleGroup', props);
	}

	export function getToggleGroupCtx() {
		return getContext<Required<ToggleGroupContext>>('toggleGroup');
	}
</script>

<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { ToggleGroup as ToggleGroupPrimitive } from 'bits-ui';

	let {
		class: className,
		orientation = 'horizontal',
		ref = $bindable(null),
		size = 'default',
		spacing = 0,
		value = $bindable(),
		variant = 'default',
		...restProps
	}: ToggleGroupPrimitive.RootProps &
		ToggleVariants & {
			spacing?: number;
			orientation?: 'horizontal' | 'vertical';
		} = $props();

	setToggleGroupCtx({
		get orientation() {
			return orientation;
		},
		get size() {
			return size;
		},
		get spacing() {
			return spacing;
		},
		get variant() {
			return variant;
		}
	});
</script>

<!--
Discriminated Unions + Destructing (required for bindable) do not
get along, so we shut typescript up by casting `value` to `never`.
-->
<ToggleGroupPrimitive.Root
	bind:ref
	bind:value={value as never}
	class={cn(
		'rounded-lg data-[size=sm]:rounded-[min(var(--radius-md),10px)] group/toggle-group flex w-fit flex-row items-center gap-[--spacing(var(--gap))] data-vertical:flex-col data-vertical:items-stretch',
		className
	)}
	data-size={size}
	data-slot="toggle-group"
	data-spacing={spacing}
	data-variant={variant}
	{orientation}
	style={`--gap: ${spacing}`}
	{...restProps}
/>
