<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ModelsSelectorList from '#lib/components/app/models/ModelsSelector/ModelsSelectorList.svelte';
	import type { GroupedModelOptions, ModelItem } from '#lib/utils/index.ts';

	interface Props {
		currentModel: string | null;
		favorites?: ModelItem[];
		groups: GroupedModelOptions;
		loaded?: ModelItem[];
	}

	let { currentModel, favorites = [], groups, loaded = [] }: Props = $props();
</script>

<!-- The rows mount tooltips, which need the provider +layout.svelte supplies. -->
<Tooltip.Provider>
	<ModelsSelectorList
		activeId={null}
		{currentModel}
		{favorites}
		{groups}
		{loaded}
		onSelect={() => {}}
	/>
</Tooltip.Provider>
