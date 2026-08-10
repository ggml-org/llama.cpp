<script lang="ts">
	import { Collapsible as CollapsiblePrimitive } from 'bits-ui';

	let {
		open = $bindable(false),
		ref = $bindable(null),
		...restProps
	}: CollapsiblePrimitive.RootProps = $props();
</script>

<CollapsiblePrimitive.Root bind:ref bind:open data-slot="collapsible" {...restProps} />
