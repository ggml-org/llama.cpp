<script lang="ts">
	import { Popover as PopoverPrimitive } from 'bits-ui';

	let { ...restProps }: PopoverPrimitive.PortalProps = $props();
</script>

<PopoverPrimitive.Portal {...restProps} />
