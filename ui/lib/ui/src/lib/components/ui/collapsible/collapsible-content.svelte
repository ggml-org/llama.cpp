<script lang="ts">
	import CollapsibleRegion from './collapsible-region.svelte';
	import { Collapsible as CollapsiblePrimitive } from 'bits-ui';

	type Props = Omit<CollapsiblePrimitive.ContentProps, 'child' | 'forceMount' | 'hiddenUntilFound'>;

	let { children, ref = $bindable(null), ...restProps }: Props = $props();
</script>

<!-- Force-mounted so the trigger keeps a real `aria-controls` target and the region
     can animate the height both ways; bits-ui's own conditional rendering would
     snap it open and shut. -->
<CollapsiblePrimitive.Content bind:ref data-slot="collapsible-content" forceMount {...restProps}>
	{#snippet child({ open, props })}
		<CollapsibleRegion {...props} {open}>
			{@render children?.()}
		</CollapsibleRegion>
	{/snippet}
</CollapsiblePrimitive.Content>
