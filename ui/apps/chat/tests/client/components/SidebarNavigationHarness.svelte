<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import SidebarNavigation from '#lib/components/app/navigation/SidebarNavigation/SidebarNavigation.svelte';
</script>

<!-- the sidebar with the tooltip provider +layout.svelte gives it -->
<Tooltip.Provider>
	<SidebarNavigation />
</Tooltip.Provider>
