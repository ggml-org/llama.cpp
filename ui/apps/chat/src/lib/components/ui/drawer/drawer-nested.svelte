<script lang="ts">
	import { Drawer as DrawerPrimitive } from 'vaul-svelte';

	let {
		activeSnapPoint = $bindable(null),
		open = $bindable(false),
		shouldScaleBackground = true,
		...restProps
	}: DrawerPrimitive.RootProps = $props();
</script>

<DrawerPrimitive.NestedRoot bind:activeSnapPoint bind:open {shouldScaleBackground} {...restProps} />
