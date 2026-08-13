<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import { Dialog as DialogPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: DialogPrimitive.TitleProps = $props();
</script>

<DialogPrimitive.Title
	bind:ref
	data-slot="dialog-title"
	class={cn('text-lg leading-none font-semibold', className)}
	{...restProps}
/>
