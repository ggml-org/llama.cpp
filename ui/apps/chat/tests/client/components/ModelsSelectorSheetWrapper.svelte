<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ModelsSelectorSheet from '#lib/components/app/models/ModelsSelector/ModelsSelectorSheet.svelte';
	import type { ComponentProps } from 'svelte';

	let props: ComponentProps<typeof ModelsSelectorSheet> = $props();
</script>

<!-- The held-row menu mounts a tooltip, which needs the provider +layout.svelte supplies. -->
<Tooltip.Provider>
	<ModelsSelectorSheet {...props} />
</Tooltip.Provider>
