/**
 * Active model resolution and capability detection for the ChatScreen.
 *
 * Picks the model that should be used for the current view
 * (router: user-selected or conversation fallback; non-router: first
 * available option), and reactively tracks which modalities (vision /
 * audio / video) it supports — fetching model props from the server on
 * demand if they aren't cached yet.
 */

import { chatStore, conversationsStore, modelsStore, serverStore } from '$lib/stores';

export function useChatScreenActiveModel() {
	const isRouter = $derived(serverStore.isRouterMode);
	const conversationModel = $derived(
		chatStore.getConversationModel(conversationsStore.activeMessages as DatabaseMessage[])
	);
	const activeModelId = $derived.by(() => {
		const options = modelsStore.models;

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = modelsStore.selectedModelId;

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
		get activeModelId() {
			return activeModelId;
		},
		get conversationModel() {
			return conversationModel;
		},
		get hasAudioModality() {
			return hasAudioModality;
		},
		get hasVideoModality() {
			return hasVideoModality;
		},
		get hasVisionModality() {
			return hasVisionModality;
		},
		get isRouter() {
			return isRouter;
		}
	};
}
