<script lang="ts">
	import { ModelBadge, ModelsSelectorDropdown } from '$lib/components/app';
	import { copyToClipboard } from '$lib/utils';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { ServerModelStatus } from '$lib/enums';

	interface Props {
		displayedModel: string | null;
		isRouter: boolean;
		isLoading: boolean;
		onRegenerate: (modelOverride?: string) => void;
	}

	let { displayedModel, isRouter, isLoading, onRegenerate }: Props = $props();

	let pendingModel = $state<string | null>(null);

	function handleCopyModel() {
		void copyToClipboard(displayedModel ?? '');
	}
</script>

{#if isRouter}
	<ModelsSelectorDropdown
		currentModel={pendingModel ?? displayedModel}
		disabled={isLoading}
		onModelChange={async (modelId: string, modelName: string) => {
			const status = modelsStore.getModelStatus(modelId);

			if (status !== ServerModelStatus.LOADED) {
				pendingModel = modelId;

				try {
					await modelsStore.loadModel(modelId);
				} finally {
					pendingModel = null;
				}
			}

			onRegenerate(modelName);
			return true;
		}}
	/>
{:else}
	<ModelBadge model={displayedModel || undefined} onclick={handleCopyModel} />
{/if}
