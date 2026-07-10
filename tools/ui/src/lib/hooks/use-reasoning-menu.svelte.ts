import { ReasoningEffort } from '$lib/enums';
import { REASONING_EFFORT_LEVELS } from '$lib/constants/reasoning-effort';
import { REASONING_EFFORT_TOKENS } from '$lib/constants/reasoning-effort-tokens';
import type { ReasoningEffortLevel } from '$lib/types';
import type { DatabaseMessage } from '$lib/types/database';
import {
	modelsStore,
	checkModelSupportsThinking,
	supportsThinking,
	propsCacheVersion,
	loadedModelIds
} from '$lib/stores/models.svelte';
import { chatStore } from '$lib/stores/chat.svelte';
import { conversationsStore, activeMessages } from '$lib/stores/conversations.svelte';
import { isRouterMode } from '$lib/stores/server.svelte';

export interface UseReasoningMenuReturn {
	readonly modelSupportsThinking: boolean;
	readonly thinkingEnabled: boolean;
	readonly currentEffort: ReasoningEffort;
	readonly levels: ReasoningEffortLevel[];
	isSelected(level: ReasoningEffortLevel): boolean;
	tokenLabel(level: ReasoningEffortLevel): string | null;
	select(level: ReasoningEffortLevel): void;
}

/**
 * Shared reactive state and helpers for the reasoning effort menu.
 *
 * Used by both the desktop dropdown (`ChatFormActionAddReasoningSubmenu`)
 * and the mobile sheet (`ChatFormActionAddSheet`) to avoid duplicating the
 * thinking-support derivation and the effort selection logic.
 */
export function useReasoningMenu(): UseReasoningMenuReturn {
	const conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);

	// a router chat can carry reasoning from an earlier turn before the props
	// cache is primed, so a model that already produced thinking still qualifies
	const modelSupportsThinkingFromMessages = $derived.by(() => {
		const modelId = isRouterMode() ? modelsStore.selectedModelName || conversationModel : null;
		if (!modelId) return false;

		return conversationsStore.activeMessages.some(
			(m) => m.role === 'assistant' && m.model === modelId && !!m.reasoningContent
		);
	});

	const modelSupportsThinking = $derived.by(() => {
		loadedModelIds();
		propsCacheVersion();

		if (isRouterMode()) {
			const modelId = modelsStore.selectedModelName || conversationModel;
			return checkModelSupportsThinking(modelId ?? '') || modelSupportsThinkingFromMessages;
		}

		return supportsThinking() || modelSupportsThinkingFromMessages;
	});

	const thinkingEnabled = $derived(conversationsStore.getThinkingEnabled());
	const currentEffort = $derived(conversationsStore.getReasoningEffort());

	return {
		get modelSupportsThinking() {
			return modelSupportsThinking;
		},
		get thinkingEnabled() {
			return thinkingEnabled;
		},
		get currentEffort() {
			return currentEffort;
		},
		get levels() {
			return REASONING_EFFORT_LEVELS;
		},
		isSelected(level: ReasoningEffortLevel): boolean {
			if (level.isOff) return !thinkingEnabled;
			return thinkingEnabled && currentEffort === level.value;
		},
		tokenLabel(level: ReasoningEffortLevel): string | null {
			if (level.isOff) return null;
			const tokens = REASONING_EFFORT_TOKENS[level.value];
			return tokens === -1 ? 'Unlimited' : `Max ${tokens.toLocaleString()} tokens`;
		},
		select(level: ReasoningEffortLevel): void {
			if (level.isOff) {
				conversationsStore.setThinkingEnabled(false);
				return;
			}
			conversationsStore.setThinkingEnabled(true);
			conversationsStore.setReasoningEffort(level.value as ReasoningEffort);
		}
	};
}
