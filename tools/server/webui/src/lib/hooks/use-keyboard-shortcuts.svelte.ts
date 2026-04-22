import { goto } from '$app/navigation';
import { KeyboardKey } from '$lib/enums';

interface KeyboardShortcutsCallbacks {
	activateSearchMode?: () => void;
	editActiveConversation?: () => void;
	onSearchActivated?: () => void;
	deleteActiveConversation?: () => void;
}

export function useKeyboardShortcuts(callbacks: KeyboardShortcutsCallbacks) {
	function handleKeydown(event: KeyboardEvent) {
		const isCtrlOrCmd = event.ctrlKey || event.metaKey;

		if (isCtrlOrCmd && event.key === KeyboardKey.K_LOWER) {
			event.preventDefault();
			callbacks.activateSearchMode?.();
			callbacks.onSearchActivated?.();
		}

		if (isCtrlOrCmd && event.shiftKey && event.key === KeyboardKey.O_UPPER) {
			event.preventDefault();
			goto('?new_chat=true#/');
		}

		if (event.shiftKey && isCtrlOrCmd && event.key === KeyboardKey.E_UPPER) {
			event.preventDefault();
			callbacks.editActiveConversation?.();
		}

		if (
			isCtrlOrCmd &&
			event.shiftKey &&
			(event.key === KeyboardKey.D_LOWER || event.key === KeyboardKey.D_UPPER)
		) {
			event.preventDefault();
			callbacks.deleteActiveConversation?.();
		}
	}

	return { handleKeydown };
}
