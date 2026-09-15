<script lang="ts">
	import { ModelBadge, ModelsSelectorDropdown } from '$lib/components/app';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore, serverStore } from '$lib/stores';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		displayedModel: string | null;
		isLoading: boolean;
		onRegenerate: (modelOverride?: string) => void;
	}

	let { displayedModel, isLoading, onRegenerate }: Props = $props();

	// same selectability rule as the form selector: router mode, or any backend
	// that exposes a selectable model list
	let isSelectable = $derived(serverStore.isRouterMode || !serverStore.capabilities.props);
	let canLoadModels = $derived(serverStore.capabilities.loadUnload);

	let pendingModel = $state<string | null>(null);

	function handleCopyModel() {
		void copyToClipboard(displayedModel ?? '');
	}
</script>

{#if isSelectable}
	<ModelsSelectorDropdown
		currentModel={pendingModel ?? displayedModel}
		disabled={isLoading}
		onModelChange={async (modelId: string, modelName: string) => {
			const status = modelsStore.getModelStatus(modelId);

			// external backends load the model implicitly on the request
			if (canLoadModels && status !== ServerModelStatus.LOADED) {
				pendingModel = modelId;

				try {
					await modelsStore.status.load(modelId);
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
