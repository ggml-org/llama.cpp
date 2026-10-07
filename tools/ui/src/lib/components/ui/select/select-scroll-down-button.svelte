<script lang="ts">
	import ChevronDownIcon from '@lucide/svelte/icons/chevron-down';
	import { cn, type WithoutChildrenOrChild } from '$lib/components/ui/utils.js';
	import { Select as SelectPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: WithoutChildrenOrChild<SelectPrimitive.ScrollDownButtonProps> = $props();
</script>

<SelectPrimitive.ScrollDownButton
	bind:ref
	class={cn('flex cursor-default items-center justify-center py-1', className)}
	data-slot="select-scroll-down-button"
	{...restProps}
>
	<ChevronDownIcon class="size-4" />
</SelectPrimitive.ScrollDownButton>
