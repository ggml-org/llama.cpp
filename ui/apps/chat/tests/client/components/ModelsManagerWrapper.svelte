<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ModelsManager from '#lib/components/app/models/ModelsManager/ModelsManager.svelte';
</script>

<!-- The rows mount tooltips, which need the provider +layout.svelte supplies. -->
<Tooltip.Provider>
	<ModelsManager />
</Tooltip.Provider>
