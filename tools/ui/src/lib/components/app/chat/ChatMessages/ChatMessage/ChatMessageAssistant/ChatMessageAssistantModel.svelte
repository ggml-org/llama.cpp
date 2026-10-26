<script lang="ts">
	import { ModelBadge, ModelsSelector } from '$lib/components/app';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		displayedModel: string | null;
		isRouter: boolean;
		isLoading: boolean;
		onRegenerate: (modelOverride?: string) => void;
	}

	let { displayedModel, isLoading, isRouter, onRegenerate }: Props = $props();

	function handleCopyModel() {
		void copyToClipboard(displayedModel ?? '');
	}
</script>

{#if isRouter}
	<ModelsSelector
		currentModel={displayedModel}
		disabled={isLoading}
		onModelChange={async (modelId: string, modelName: string) => {
			const status = modelsStore.getModelStatus(modelId);

			// the picker's own selection applies before this resolves, so a model
			// that is not loaded yet gets its load request in first
			if (status !== ServerModelStatus.LOADED) {
				await modelsStore.status.load(modelId);
			}

			onRegenerate(modelName);

			return true;
		}}
	/>
{:else}
	<ModelBadge model={displayedModel || undefined} onclick={handleCopyModel} />
{/if}
