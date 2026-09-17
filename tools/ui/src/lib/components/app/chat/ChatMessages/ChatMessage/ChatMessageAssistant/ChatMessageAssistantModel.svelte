<script lang="ts">
	import { ModelBadge, ModelsSelectorDropdown } from '$lib/components/app';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore, serverStore } from '$lib/stores';
	import { copyToClipboard, getBackendCapabilities } from '$lib/utils';
	import { getBackend } from '$lib/utils/api-base';

	interface Props {
		displayedModel: string | null;
		isLoading: boolean;
		onRegenerate: (modelOverride?: string) => void;
	}

	let { displayedModel, isLoading, onRegenerate }: Props = $props();

	// same selectability rule as the form selector: router mode, or any backend
	// that exposes a selectable model list
	let isSelectable = $derived(serverStore.isRouterMode || !serverStore.capabilities.props);

	let pendingModel = $state<string | null>(null);

	function handleCopyModel() {
		void copyToClipboard(displayedModel ?? '');
	}
</script>

{#if isSelectable}
	<ModelsSelectorDropdown
		currentModel={pendingModel ?? displayedModel}
		disabled={isLoading}
		onModelChange={async (modelId: string, modelName: string, backendId?: string) => {
			// capability of the picked model's own backend, not the active one
			const loadsOnRequest = getBackendCapabilities(getBackend(backendId)).loadUnload;
			const status = modelsStore.getModelStatus(modelId);

			// only a llama.cpp server loads up front; remote backends load the
			// model with the request itself
			if (loadsOnRequest && status !== ServerModelStatus.LOADED) {
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
