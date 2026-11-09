import { page } from '$app/state';
import { NEW_CHAT_TAB_ID } from '$lib/constants';
import { KeyboardKey } from '$lib/enums';
import { conversationsStore, settingsStore, tabsStore } from '$lib/stores';

interface KeyboardShortcutsCallbacks {
	editActiveConversation?: () => void;
	deleteActiveConversation?: () => void;
	navigateToPrevConversation?: () => void;
	navigateToNextConversation?: () => void;
	navigateToPrevTab?: () => void;
	navigateToNextTab?: () => void;
	openModelsManager?: () => void;
	openSearch?: () => void;
	openSettings?: () => void;
	toggleSidebar?: () => void;
}

export function useKeyboardShortcuts(callbacks: KeyboardShortcutsCallbacks) {
	function handleKeydown(event: KeyboardEvent) {
		const isCmdOrCtrl = event.metaKey || event.ctrlKey;

		if (isCmdOrCtrl && event.key === KeyboardKey.K_LOWER) {
			event.preventDefault();
			callbacks.openSearch?.();
		}

		if (isCmdOrCtrl && event.key === KeyboardKey.COMMA) {
			event.preventDefault();
			callbacks.openSettings?.();
		}

		if (isCmdOrCtrl && event.key === KeyboardKey.B_LOWER) {
			event.preventDefault();
			callbacks.toggleSidebar?.();
		}

		if (
			isCmdOrCtrl &&
			event.shiftKey &&
			(event.key === KeyboardKey.O_LOWER || event.key === KeyboardKey.O_UPPER)
		) {
			event.preventDefault();

			void conversationsStore.openNewChat();
		}

		if (
			isCmdOrCtrl &&
			event.shiftKey &&
			(event.key === KeyboardKey.M_LOWER || event.key === KeyboardKey.M_UPPER)
		) {
			event.preventDefault();
			callbacks.openModelsManager?.();
		}

		if (event.shiftKey && isCmdOrCtrl && event.key === KeyboardKey.E_UPPER) {
			event.preventDefault();
			callbacks.editActiveConversation?.();
		}

		if (
			event.shiftKey &&
			isCmdOrCtrl &&
			(event.key === KeyboardKey.X_LOWER || event.key === KeyboardKey.X_UPPER)
		) {
			// several components register this shortcut; only let the first handler
			// act so the synchronous navigation does not cascade-close every tab
			if (event.defaultPrevented) return;

			// close-tab only makes sense with conversation tabs enabled
			if (!settingsStore.config.conversationTabs) return;

			event.preventDefault();

			const activeId =
				page.params.id ?? (page.route.id === '/(chat)' ? NEW_CHAT_TAB_ID : undefined);

			if (activeId) {
				void tabsStore.close(activeId, true);
			}
		}

		if (
			isCmdOrCtrl &&
			event.shiftKey &&
			(event.key === KeyboardKey.D_LOWER || event.key === KeyboardKey.D_UPPER)
		) {
			event.preventDefault();
			callbacks.deleteActiveConversation?.();
		}

		if (isCmdOrCtrl && event.shiftKey && event.key === KeyboardKey.ARROW_UP) {
			event.preventDefault();
			callbacks.navigateToPrevConversation?.();
		}

		if (isCmdOrCtrl && event.shiftKey && event.key === KeyboardKey.ARROW_DOWN) {
			event.preventDefault();
			callbacks.navigateToNextConversation?.();
		}

		if (isCmdOrCtrl && event.altKey && event.shiftKey && event.code === KeyboardKey.BRACKET_LEFT) {
			event.preventDefault();
			callbacks.navigateToPrevTab?.();
		}

		if (isCmdOrCtrl && event.altKey && event.shiftKey && event.code === KeyboardKey.BRACKET_RIGHT) {
			event.preventDefault();
			callbacks.navigateToNextTab?.();
		}
	}

	return { handleKeydown };
}
