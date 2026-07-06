/**
 * Active model resolution and capability detection for the ChatScreen.
 *
 * Picks the model that should be used for the current view
 * (router: user-selected or conversation fallback; non-router: first
 * available option), and reactively tracks which modalities (vision /
 * audio / video) it supports — fetching model props from the server on
 * demand if they aren't cached yet.
 */

import { modelsStore, modelOptions, selectedModelId } from '$lib/stores/models.svelte';
import { isRouterMode } from '$lib/stores/server.svelte';
import { chatStore } from '$lib/stores/chat.svelte';
import { activeMessages } from '$lib/stores/conversations.svelte';

export function useChatScreenActiveModel() {
	const isRouter = $derived(isRouterMode());
	const conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);

	const activeModelId = $derived.by(() => {
		const options = modelOptions();

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = selectedModelId();
		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);
			if (model) return model.model;
		}

		return null;
	});

	let modelPropsVersion = $state(0);

	$effect(() => {
		if (activeModelId) {
			const cached = modelsStore.getModelProps(activeModelId);
			if (!cached) {
				modelsStore.fetchModelProps(activeModelId).then(() => {
					modelPropsVersion++;
				});
			}
		}
	});

	const hasAudioModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;
			return modelsStore.modelSupportsAudio(activeModelId);
		}
		return false;
	});

	const hasVideoModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;
			return modelsStore.modelSupportsVideo(activeModelId);
		}
		return false;
	});

	const hasVisionModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;
			return modelsStore.modelSupportsVision(activeModelId);
		}
		return false;
	});

	return {
		get isRouter() {
			return isRouter;
		},
		get conversationModel() {
			return conversationModel;
		},
		get activeModelId() {
			return activeModelId;
		},
		get hasAudioModality() {
			return hasAudioModality;
		},
		get hasVideoModality() {
			return hasVideoModality;
		},
		get hasVisionModality() {
			return hasVisionModality;
		}
	};
}
