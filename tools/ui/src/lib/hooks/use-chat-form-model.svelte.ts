/**
 * Model state for the chat form's action bar.
 *
 * Syncs the conversation's model into the global selection (a conversation
 * opened from history names the model that produced it), falls back to the
 * first loaded model on a router, and reactively tracks which modalities
 * (vision / audio / video) the active model supports - fetching model props
 * from the server on demand if they are not cached yet.
 */

import { conversationsStore, modelsStore, serverStore } from '$lib/stores';
import type { DatabaseMessage } from '$lib/types/database';
import { getConversationModel } from '$lib/utils';

export function useChatFormModel() {
	const isRouter = $derived(serverStore.isRouterMode);
	const conversationModel = $derived(
		getConversationModel(conversationsStore.activeMessages as DatabaseMessage[])
	);

	let lastSyncedConversationModel: string | null = null;

	// the conversation's model wins while it names one: opening a history chat
	// points the selection at the model that produced it
	$effect(() => {
		if (conversationModel && conversationModel !== lastSyncedConversationModel) {
			if (modelsStore.models.some((m) => m.model === conversationModel)) {
				modelsStore.selectedModelName = conversationModel;
				modelsStore.selectModelByName(conversationModel);
			} else {
				modelsStore.selectedModelName = null;
				modelsStore.clearSelection();
			}

			lastSyncedConversationModel = conversationModel;
		} else if (
			isRouter &&
			!modelsStore.selectedModelId &&
			modelsStore.loadedModelIds.length > 0 &&
			conversationsStore.activeMessages.length > 0 &&
			!conversationModel
		) {
			lastSyncedConversationModel = null;
			const first = modelsStore.models.find((m) => modelsStore.loadedModelIds.includes(m.model));

			if (first) modelsStore.selectModelById(first.id);
		}
	});

	const activeModelId = $derived(modelsStore.activeModelId);

	let modelPropsVersion = $state(0); // Used to trigger reactivity after fetch

	$effect(() => {
		if (activeModelId) {
			const cached = modelsStore.props.getModelProps(activeModelId);

			if (!cached) {
				modelsStore.props.fetchModelProps(activeModelId).then(() => {
					modelPropsVersion++;
				});
			}
		}
	});

	const hasAudioModality = $derived.by(() => {
		void modelPropsVersion;

		return activeModelId ? modelsStore.props.modelSupportsAudio(activeModelId) : false;
	});
	const hasVideoModality = $derived.by(() => {
		void modelPropsVersion;

		return activeModelId ? modelsStore.props.modelSupportsVideo(activeModelId) : false;
	});
	const hasVisionModality = $derived.by(() => {
		void modelPropsVersion;

		return activeModelId ? modelsStore.props.modelSupportsVision(activeModelId) : false;
	});
	const hasModelSelected = $derived(
		!isRouter || !!conversationModel || !!modelsStore.selectedModelId
	);
	const isSelectedModelInCache = $derived.by(() => {
		if (!isRouter) return true;

		if (conversationModel) {
			return modelsStore.models.some((option) => option.model === conversationModel);
		}

		const currentModelId = modelsStore.selectedModelId;

		if (!currentModelId) return false;

		return modelsStore.models.some((option) => option.id === currentModelId);
	});

	return {
		get conversationModel() {
			return conversationModel;
		},
		get hasAudioModality() {
			return hasAudioModality;
		},
		get hasModelSelected() {
			return hasModelSelected;
		},
		get hasVideoModality() {
			return hasVideoModality;
		},
		get hasVisionModality() {
			return hasVisionModality;
		},
		get isRouter() {
			return isRouter;
		},
		get isSelectedModelInCache() {
			return isSelectedModelInCache;
		}
	};
}
