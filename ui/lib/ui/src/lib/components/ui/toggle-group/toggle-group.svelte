<script lang="ts" module>
	import type { ToggleSize } from '$lib/components/ui/toggle/index.js';
	import { getContext, setContext } from 'svelte';

	/** A group is as tall as a control of its size; its keys fill it. */
	const GROUP_HEIGHT: Record<NonNullable<ToggleSize>, string> = {
		default: 'data-[orientation=horizontal]:h-control-md',
		icon: 'data-[orientation=horizontal]:h-control-md',
		'icon-lg': 'data-[orientation=horizontal]:h-control-lg',
		'icon-sm': 'data-[orientation=horizontal]:h-control-sm',
		lg: 'data-[orientation=horizontal]:h-control-lg',
		sm: 'data-[orientation=horizontal]:h-control-sm'
	};

	interface ToggleGroupContext {
		size?: ToggleSize;
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
		size = 'sm',
		value = $bindable(),
		...restProps
	}: ToggleGroupPrimitive.RootProps & {
		size?: ToggleSize;
		orientation?: 'horizontal' | 'vertical';
	} = $props();

	setToggleGroupCtx({
		get orientation() {
			return orientation;
		},
		get size() {
			return size;
		}
	});
</script>

<!-- a row of keys like a secondary button, an outline-coloured seam between them -->
<!--
Discriminated Unions + Destructing (required for bindable) do not
get along, so we shut typescript up by casting `value` to `never`.
-->
<ToggleGroupPrimitive.Root
	bind:ref
	bind:value={value as never}
	class={cn(
		'group/toggle-group flex w-fit flex-row items-stretch overflow-hidden rounded-control border border-(--toggle-group-border) bg-(--toggle-group-background) shadow-(--toggle-group-shadow) matte-key data-vertical:flex-col',
		GROUP_HEIGHT[size ?? 'sm'],
		className
	)}
	data-size={size}
	data-slot="toggle-group"
	{orientation}
	{...restProps}
/>
