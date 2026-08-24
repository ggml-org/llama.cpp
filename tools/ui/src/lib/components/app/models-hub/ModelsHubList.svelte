<script lang="ts">
	import type { HfModelInfo } from '$lib/types/huggingface';
	import ModelsHubListItem from './ModelsHubListItem.svelte';

	interface Props {
		models: HfModelInfo[];
		activeId?: string | null;
		/** Show the original (base) model's org avatar instead of the repo's org. */
		showBaseModelAvatar?: boolean;
		onSelect?: (modelId: string) => void;
	}

	let { models, activeId = null, showBaseModelAvatar = false, onSelect }: Props = $props();
</script>

<ul class="space-y-0.5 p-2">
	{#each models as model (model.id)}
		<ModelsHubListItem {model} active={model.id === activeId} {showBaseModelAvatar} {onSelect} />
	{/each}
</ul>
