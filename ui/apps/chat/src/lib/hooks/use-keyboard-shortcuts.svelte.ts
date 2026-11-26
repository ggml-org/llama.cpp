import { KeyboardKey } from '#lib/enums/index.ts';
import { conversationsStore, deviceStore, settingsStore, tabsStore } from '#lib/stores/index.ts';
import { resolveTabTarget, resolveWindowCloseTab } from '#lib/utils/index.ts';
import { page } from '$app/state';

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

			void conversationsStore.openNewChat({ inNewTab: true });
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

			const activeId = resolveTabTarget(page);

			if (activeId) {
				void tabsStore.close(activeId, true);
			}
		}

		// the browser's own close shortcut: Cmd+W on Apple platforms, where Ctrl+W
		// deletes a word in text fields, and Ctrl+W elsewhere
		const isWindowClose =
			(deviceStore.isApplePlatform ? event.metaKey && !event.ctrlKey : event.ctrlKey) &&
			!event.shiftKey &&
			!event.altKey &&
			event.key === KeyboardKey.W_LOWER;

		// the tab strip is hidden on mobile, where the shortcut keeps closing the window
		if (isWindowClose && settingsStore.config.conversationTabs && !deviceStore.isMobile) {
			if (event.defaultPrevented) return;

			const tabId = resolveWindowCloseTab(page, tabsStore.openTabs);

			if (tabId) {
				event.preventDefault();
				void tabsStore.close(tabId, true);
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
