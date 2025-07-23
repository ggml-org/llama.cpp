import { ChatService } from '$lib/services/chat';
import type { ChatState } from './handle-send-message';

export function createHandleStopGeneration(
	chatService: ChatService,
	updateState: (updater: (state: ChatState) => ChatState) => void
) {
	return function handleStopGeneration() {
		if (chatService) {
			chatService.abort();
			updateState((state) => ({
				...state,
				isLoading: false
			}));
		}
	};
}
