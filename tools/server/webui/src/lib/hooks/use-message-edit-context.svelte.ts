import { setMessageEditContext } from '$lib/contexts';
import { MessageRole } from '$lib/enums';
import { parseFilesToMessageExtras } from '$lib/utils/convert-files-to-extra';

interface UseMessageEditContextOptions {
	getContent: () => string;
	getExtras: () => DatabaseMessageExtra[];
	showSaveOnlyOption?: boolean;
	onSave: (content: string, extras?: DatabaseMessageExtra[]) => void;
}

export function useMessageEditContext(options: UseMessageEditContextOptions) {
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

	setMessageEditContext({
		get isEditing() {
			return isEditing;
		},
		get editedContent() {
			return editedContent;
		},
		get editedExtras() {
			return editedExtras;
		},
		get editedUploadedFiles() {
			return editedUploadedFiles;
		},
		get originalContent() {
			return options.getContent();
		},
		get originalExtras() {
			return options.getExtras();
		},
		get showSaveOnlyOption() {
			return options.showSaveOnlyOption ?? false;
		},
		get showBranchAfterEditOption() {
			return false;
		},
		get shouldBranchAfterEdit() {
			return false;
		},
		get messageRole() {
			return MessageRole.USER;
		},
		setContent: (c: string) => {
			editedContent = c;
		},
		setExtras: (e: DatabaseMessageExtra[]) => {
			editedExtras = e;
		},
		setUploadedFiles: (f: ChatUploadedFile[]) => {
			editedUploadedFiles = f;
		},
		save: handleSaveEdit,
		saveOnly: handleSaveEdit,
		cancel: handleCancelEdit,
		startEdit: handleEdit
	});

	return {
		get isEditing() {
			return isEditing;
		},
		handleEdit,
		handleSaveEdit,
		handleCancelEdit
	};
}
