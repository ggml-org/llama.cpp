<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ModelsManagerModelRow from '#lib/components/app/models/ModelsManager/ModelsManagerModelRow.svelte';
	import type { ModelOption } from '#lib/types/models.js';

	interface Props {
		option: ModelOption;
		selected?: boolean;
		onSelect: (option: ModelOption) => void;
	}

	let { onSelect, option, selected = false }: Props = $props();
</script>

<!-- The row actions mount a tooltip, which needs the provider +layout.svelte supplies. -->
<Tooltip.Provider>
	<ModelsManagerModelRow
		isFavorite={() => false}
		onDelete={() => {}}
		{onSelect}
		{option}
		{selected}
	/>
</Tooltip.Provider>
