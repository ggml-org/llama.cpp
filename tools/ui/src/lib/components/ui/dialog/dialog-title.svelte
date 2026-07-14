<script lang="ts">
	import { Dialog as DialogPrimitive } from 'bits-ui';
	import { cn } from '$lib/components/ui/utils';

	let {
		ref = $bindable(null),
		class: className,
		...restProps
	}: DialogPrimitive.TitleProps = $props();
</script>

<DialogPrimitive.Title
	bind:ref
	data-slot="dialog-title"
	class={cn('text-lg leading-none font-semibold', className)}
	{...restProps}
/>
