import {
	SETTINGS_KEYS,
	TOOL_CALL_SHOWN_ARG_NONE,
	TOOL_SETTINGS_TEXT,
	TOOL_SHOWN_ARG_PICK
} from '#lib/constants/index.ts';
import { conversationsStore, settingsStore } from '#lib/stores/index.ts';
import { parseToolCallShownArgOverrides } from '#lib/utils/index.ts';

/**
 * State of the Tools settings, for the desktop's table and the phone sheet's pages: which argument
 * each tool's calls show, and what the tool switches apply to.
 */
export function useToolSettings() {
	const shownArgOverrides = $derived(
		parseToolCallShownArgOverrides(
			settingsStore.config[SETTINGS_KEYS.TOOL_CALL_SHOWN_ARG_OVERRIDES]
		)
	);
	const globalShownArg = $derived(
		String(settingsStore.config[SETTINGS_KEYS.TOOL_CALL_SHOWN_ARG] ?? '').trim()
	);
	const scopeNote = $derived(
		conversationsStore.activeConversation
			? TOOL_SETTINGS_TEXT.SCOPE_CONVERSATION
			: TOOL_SETTINGS_TEXT.SCOPE_NEW
	);

	return {
		get scopeNote() {
			return scopeNote;
		},
		setShownArg(toolName: string, pick: string) {
			const next = { ...shownArgOverrides };

			if (pick === TOOL_SHOWN_ARG_PICK.DEFAULT) delete next[toolName];
			else next[toolName] = pick === TOOL_SHOWN_ARG_PICK.NONE ? TOOL_CALL_SHOWN_ARG_NONE : pick;

			settingsStore.updateConfig(SETTINGS_KEYS.TOOL_CALL_SHOWN_ARG_OVERRIDES, JSON.stringify(next));
		},
		shownArgLabel(pick: string): string {
			if (pick === TOOL_SHOWN_ARG_PICK.NONE) return TOOL_SETTINGS_TEXT.SHOWN_ARG_NONE;

			if (pick === TOOL_SHOWN_ARG_PICK.DEFAULT) {
				return globalShownArg
					? TOOL_SETTINGS_TEXT.SHOWN_ARG_DEFAULT_OF(globalShownArg)
					: TOOL_SETTINGS_TEXT.SHOWN_ARG_DEFAULT;
			}

			return pick;
		},
		/** A tool follows the global setting until it gets a pick of its own. */
		shownArgPick(toolName: string): string {
			const own = shownArgOverrides[toolName];

			if (own === undefined) return TOOL_SHOWN_ARG_PICK.DEFAULT;

			return own === TOOL_CALL_SHOWN_ARG_NONE ? TOOL_SHOWN_ARG_PICK.NONE : own;
		}
	};
}
