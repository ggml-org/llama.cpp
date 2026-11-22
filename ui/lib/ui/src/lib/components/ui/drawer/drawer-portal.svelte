<script lang="ts">
	import { Drawer as DrawerPrimitive } from 'vaul-svelte';

	let { ...restProps }: DrawerPrimitive.PortalProps = $props();
</script>

<DrawerPrimitive.Portal {...restProps} />
