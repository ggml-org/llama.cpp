<script lang="ts">
	import { cn } from '../utils';
	import { Dialog as DialogPrimitive } from 'bits-ui';

	let {
		class: className,
		ref = $bindable(null),
		...restProps
	}: DialogPrimitive.TitleProps = $props();
</script>

<DialogPrimitive.Title
	bind:ref
	class={cn('text-title leading-none font-semibold', className)}
	data-slot="dialog-title"
	{...restProps}
/>
