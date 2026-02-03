<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import Page from '../../../src/routes/+page.svelte';

	let sidebarOpen = $state(false);
</script>

<!--
	Test wrapper that provides necessary context providers for component testing.
	This mirrors the providers from +layout.svelte.
-->
<Tooltip.Provider>
	<Sidebar.Provider bind:open={sidebarOpen}>
		<Page />
	</Sidebar.Provider>
</Tooltip.Provider>
