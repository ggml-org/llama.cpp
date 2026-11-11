<script lang="ts">
	import SidebarNavigation from '$lib/components/app/navigation/SidebarNavigation/SidebarNavigation.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
</script>

<!-- the sidebar with the tooltip provider +layout.svelte gives it -->
<Tooltip.Provider>
	<SidebarNavigation />
</Tooltip.Provider>
