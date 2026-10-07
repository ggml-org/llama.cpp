<script lang="ts">
	import { LinkPreview as HoverCardPrimitive } from 'bits-ui';

	let { ref = $bindable(null), ...restProps }: HoverCardPrimitive.TriggerProps = $props();
</script>

<HoverCardPrimitive.Trigger bind:ref data-slot="hover-card-trigger" {...restProps} />
