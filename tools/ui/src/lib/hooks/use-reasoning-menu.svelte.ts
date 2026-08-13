import { REASONING_EFFORT_LEVELS, REASONING_EFFORT_TOKENS } from '$lib/constants';
import { ReasoningEffort } from '$lib/enums';
import { chatStore } from '$lib/stores/chat.svelte';
import { activeMessages, conversationsStore } from '$lib/stores/conversations.svelte';
import {
	checkModelSupportsThinking,
	loadedModelIds,
	modelsStore,
	propsCacheVersion,
	supportsThinking
} from '$lib/stores/models.svelte';
import { isRouterMode } from '$lib/stores/server.svelte';
import type { ReasoningEffortLevel } from '$lib/types';
import type { DatabaseMessage } from '$lib/types/database';

export interface UseReasoningMenuReturn {
	readonly modelSupportsThinking: boolean;
	readonly thinkingEnabled: boolean;
	readonly isOff: boolean;
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
	const currentEffort = $derived(conversationsStore.getReasoningEffort());
	const thinkingEnabled = $derived(
		currentEffort !== ReasoningEffort.OFF && currentEffort !== ReasoningEffort.DEFAULT
	);

	return {
		get currentEffort() {
			return currentEffort;
		},
		get isOff() {
			return currentEffort === ReasoningEffort.OFF;
		},
		isSelected(level: ReasoningEffortLevel): boolean {
			return currentEffort === level.value;
		},
		get levels() {
			return REASONING_EFFORT_LEVELS;
		},
		get modelSupportsThinking() {
			return modelSupportsThinking;
		},
		select(level: ReasoningEffortLevel): void {
			conversationsStore.setReasoningEffort(level.value as ReasoningEffort);
		},
		get thinkingEnabled() {
			return thinkingEnabled;
		},
		tokenLabel(level: ReasoningEffortLevel): string | null {
			if (level.value === ReasoningEffort.DEFAULT) return 'Model default';

			const tokens = REASONING_EFFORT_TOKENS[level.value];

			if (tokens === undefined) return null;

			return tokens === -1 ? 'Unlimited' : `Max ${tokens.toLocaleString()} tokens`;
		}
	};
}
