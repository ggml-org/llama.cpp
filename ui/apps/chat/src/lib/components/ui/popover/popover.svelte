<script lang="ts">
	import { Popover as PopoverPrimitive } from 'bits-ui';

	let { open = $bindable(false), ...restProps }: PopoverPrimitive.RootProps = $props();
</script>

<PopoverPrimitive.Root bind:open {...restProps} />
