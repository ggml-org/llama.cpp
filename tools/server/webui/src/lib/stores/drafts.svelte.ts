import { NEW_CHAT_DRAFT_KEY } from '$lib/constants';

interface Draft {
	message: string;
	files: ChatUploadedFile[];
}

class DraftsStore {
	private drafts = new Map<string, Draft>();

	getDraft(chatId: string | undefined): Draft {
		const key = chatId ?? NEW_CHAT_DRAFT_KEY;
		return this.drafts.get(key) ?? { message: '', files: [] };
	}

	saveDraft(chatId: string | undefined, message: string, files: ChatUploadedFile[]): void {
		const key = chatId ?? NEW_CHAT_DRAFT_KEY;
		if (message || files.length > 0) {
			this.drafts.set(key, { message, files: [...files] });
		} else {
			this.drafts.delete(key);
		}
	}

	clearDraft(chatId: string | undefined): void {
		const key = chatId ?? NEW_CHAT_DRAFT_KEY;
		this.drafts.delete(key);
	}
}

export const draftsStore = new DraftsStore();
