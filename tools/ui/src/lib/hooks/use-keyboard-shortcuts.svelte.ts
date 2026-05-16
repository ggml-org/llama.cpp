import { goto } from '$app/navigation';
import { KeyboardKey } from '$lib/enums';
import { ROUTES } from '$lib/constants/routes';

interface KeyboardShortcutsCallbacks {
	activateSearchMode?: () => void;
	editActiveConversation?: () => void;
	onSearchActivated?: () => void;
	deleteActiveConversation?: () => void;
	navigateToPrevConversation?: () => void;
	navigateToNextConversation?: () => void;
}

export function useKeyboardShortcuts(callbacks: KeyboardShortcutsCallbacks) {
	function handleKeydown(event: KeyboardEvent) {
		const isCmdOrCtrl = event.metaKey || event.ctrlKey;

		if (isCmdOrCtrl && event.key === KeyboardKey.K_LOWER) {
			event.preventDefault();
			callbacks.activateSearchMode?.();
			callbacks.onSearchActivated?.();
		}

		if (
			isCmdOrCtrl &&
			event.shiftKey &&
			(event.key === KeyboardKey.O_LOWER || event.key === KeyboardKey.O_UPPER)
		) {
			event.preventDefault();

			goto(ROUTES.NEW_CHAT);
		}

		if (event.shiftKey && isCmdOrCtrl && event.key === KeyboardKey.E_UPPER) {
			event.preventDefault();
			callbacks.editActiveConversation?.();
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
	}

	return { handleKeydown };
}
