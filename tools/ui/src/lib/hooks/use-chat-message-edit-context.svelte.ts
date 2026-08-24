import { setChatMessageEditContext } from '$lib/contexts';
import { MessageRole } from '$lib/enums';
import { parseFilesToMessageExtras } from '$lib/utils/convert-files-to-extra';

interface UseChatMessageEditContextOptions {
	getContent: () => string;
	getExtras: () => DatabaseMessageExtra[];
	showSaveOnlyOption?: boolean;
	onSave: (content: string, extras?: DatabaseMessageExtra[]) => void;
}

export function useChatMessageEditContext(options: UseChatMessageEditContextOptions) {
	let isEditing = $state(false);
	let editedContent = $state('');
	let editedExtras = $state<DatabaseMessageExtra[]>([]);
	let editedUploadedFiles = $state<ChatUploadedFile[]>([]);

	function handleEdit() {
		editedContent = options.getContent();
		editedExtras = [...options.getExtras()];
		editedUploadedFiles = [];
		isEditing = true;
	}

	async function handleSaveEdit() {
		const trimmed = editedContent.trim();

		if (!trimmed && editedExtras.length === 0 && editedUploadedFiles.length === 0) return;

		let finalExtras: DatabaseMessageExtra[] = $state.snapshot(editedExtras);

		if (editedUploadedFiles.length > 0) {
			const plainFiles = $state.snapshot(editedUploadedFiles);
			const result = await parseFilesToMessageExtras(plainFiles);
			const newExtras = result?.extras || [];

			finalExtras = [...finalExtras, ...newExtras];
		}

		options.onSave(trimmed, finalExtras.length > 0 ? finalExtras : undefined);
		isEditing = false;
	}

	function handleCancelEdit() {
		isEditing = false;
	}

	setChatMessageEditContext({
		cancel: handleCancelEdit,
		get editedContent() {
			return editedContent;
		},
		get editedExtras() {
			return editedExtras;
		},
		// A pending user message carries no reasoning, so the edit form stays single field
		get editedReasoning() {
			return '';
		},
		get editedUploadedFiles() {
			return editedUploadedFiles;
		},
		get isEditing() {
			return isEditing;
		},
		get messageRole() {
			return MessageRole.USER;
		},
		get originalContent() {
			return options.getContent();
		},
		get originalExtras() {
			return options.getExtras();
		},
		get originalReasoning() {
			return '';
		},
		save: handleSaveEdit,
		saveOnly: handleSaveEdit,
		setContent: (c: string) => {
			editedContent = c;
		},
		setExtras: (e: DatabaseMessageExtra[]) => {
			editedExtras = e;
		},
		setUploadedFiles: (f: ChatUploadedFile[]) => {
			editedUploadedFiles = f;
		},
		get shouldBranchAfterEdit() {
			return false;
		},
		get showBranchAfterEditOption() {
			return false;
		},
		get showReasoningField() {
			return false;
		},
		get showSaveOnlyOption() {
			return options.showSaveOnlyOption ?? false;
		},
		startEdit: handleEdit
	});

	return {
		handleCancelEdit,
		handleEdit,
		handleSaveEdit,
		get isEditing() {
			return isEditing;
		}
	};
}
