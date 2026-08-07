import { NEW_CHAT_DRAFT_KEY } from '$lib/constants';

interface DraftMessage {
	message: string;
	files: ChatUploadedFile[];
}

class DraftMessagesStore {
	private drafts = new Map<string, DraftMessage>();

	getDraftMessage(chatId: string | undefined): DraftMessage {
		const key = chatId ?? NEW_CHAT_DRAFT_KEY;
		return this.drafts.get(key) ?? { message: '', files: [] };
	}

	saveDraftMessage(chatId: string | undefined, message: string, files: ChatUploadedFile[]): void {
		const key = chatId ?? NEW_CHAT_DRAFT_KEY;
		if (message || files.length > 0) {
			this.drafts.set(key, { message, files: [...files] });
		} else {
			this.drafts.delete(key);
		}
	}

	clearDraftMessage(chatId: string | undefined): void {
		const key = chatId ?? NEW_CHAT_DRAFT_KEY;
		this.drafts.delete(key);
	}
}

export const draftMessagesStore = new DraftMessagesStore();
