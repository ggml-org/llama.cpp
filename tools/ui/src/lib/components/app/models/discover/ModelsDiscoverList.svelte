<script lang="ts">
	import ModelsDiscoverItem from './ModelsDiscoverItem.svelte';
	import type { HfModelInfo } from '$lib/types/huggingface';

	interface Props {
		models: HfModelInfo[];
		activeId?: string | null;
		/** Show the original (base) model's org avatar instead of the repo's org. */
		showBaseModelAvatar?: boolean;
		onSelect?: (modelId: string) => void;
	}

	let { activeId = null, models, onSelect, showBaseModelAvatar = false }: Props = $props();
</script>

<ul class="space-y-0.5 p-2">
	{#each models as model (model.id)}
		<ModelsDiscoverItem active={model.id === activeId} {model} {onSelect} {showBaseModelAvatar} />
	{/each}
</ul>
