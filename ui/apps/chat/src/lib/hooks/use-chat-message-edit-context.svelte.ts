import { setChatMessageEditContext } from '#lib/contexts/index.ts';
import { MessageRole } from '#lib/enums/index.ts';

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

	function handleSaveEdit() {
		const trimmed = editedContent.trim();

		if (!trimmed && editedExtras.length === 0 && editedUploadedFiles.length === 0) return;

		options.onSave(trimmed, editedExtras.length > 0 ? $state.snapshot(editedExtras) : undefined);
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
