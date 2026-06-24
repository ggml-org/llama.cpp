<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import Page from '../../../src/routes/(chat)/+page.svelte';
</script>

<!--
	Test wrapper that provides necessary context providers for component testing.
	This mirrors the providers from +layout.svelte.
-->
<Tooltip.Provider>
	<Page />
</Tooltip.Provider>
