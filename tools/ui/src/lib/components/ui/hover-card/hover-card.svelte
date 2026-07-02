<script lang="ts">
	import { LinkPreview as HoverCardPrimitive } from "bits-ui";

	let { open = $bindable(false), ...restProps }: HoverCardPrimitive.RootProps = $props();
</script>

<HoverCardPrimitive.Root bind:open {...restProps} />
