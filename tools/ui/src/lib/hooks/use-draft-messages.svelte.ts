import { onMount } from 'svelte';
import { afterNavigate, beforeNavigate } from '$app/navigation';
import { draftMessagesStore } from '$lib/stores/draft-messages.svelte';

interface UseDraftMessagesOptions {
	getChatId: () => string | undefined;
	getMessage: () => string;
	getFiles: () => ChatUploadedFile[];
	setMessage: (message: string) => void;
	setFiles: (files: ChatUploadedFile[]) => void;
	getInitialMessage: () => string;
}

export function useDraftMessages(options: UseDraftMessagesOptions) {
	onMount(() => {
		const chatId = options.getChatId();
		const draft = draftMessagesStore.getDraftMessage(chatId);

		if ((draft.message || draft.files.length > 0) && !options.getInitialMessage()) {
			options.setMessage(draft.message);
			options.setFiles(draft.files);
		}
	});

	beforeNavigate(() => {
		const chatId = options.getChatId();
		draftMessagesStore.saveDraftMessage(chatId, options.getMessage(), options.getFiles());
	});

	afterNavigate((navigation) => {
		if (navigation?.from != null) {
			const chatId = options.getChatId();
			const draft = draftMessagesStore.getDraftMessage(chatId);
			options.setMessage(draft.message);
			options.setFiles(draft.files);
		}
	});

	function clearDraft() {
		const chatId = options.getChatId();
		draftMessagesStore.clearDraftMessage(chatId);
	}

	return { clearDraft };
}
