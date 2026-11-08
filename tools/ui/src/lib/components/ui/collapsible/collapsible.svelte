<script lang="ts">
	import { Collapsible as CollapsiblePrimitive } from 'bits-ui';

	let {
		open = $bindable(false),
		ref = $bindable(null),
		...restProps
	}: CollapsiblePrimitive.RootProps = $props();
</script>

<CollapsiblePrimitive.Root bind:open bind:ref data-slot="collapsible" {...restProps} />
