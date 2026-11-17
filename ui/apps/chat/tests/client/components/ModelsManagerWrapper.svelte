<script lang="ts">
	import ModelsManager from '#lib/components/app/models/ModelsManager/ModelsManager.svelte';
	import * as Tooltip from '#lib/components/ui/tooltip/index.ts';
</script>

<!-- The rows mount tooltips, which need the provider +layout.svelte supplies. -->
<Tooltip.Provider>
	<ModelsManager />
</Tooltip.Provider>
