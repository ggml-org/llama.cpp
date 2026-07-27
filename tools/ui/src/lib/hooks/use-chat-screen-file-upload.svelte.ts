/**
 * File upload lifecycle for the ChatScreen form.
 *
 * Owns the queue of processed `ChatUploadedFile`, the rejection-by-capability
 * dialog state, and the dual-layer validation pipeline (general format +
 * model modality). The caller provides the active model's capabilities and ID
 * as reactive getters so validation tracks the model in real time.
 */

import { processFilesToChatUploaded } from '$lib/utils/browser-only';
import { isFileTypeSupported, filterFilesByModalities } from '$lib/utils';

interface UseChatScreenFileUploadOptions {
	capabilities: () => { hasVision: boolean; hasAudio: boolean; hasVideo: boolean };
	activeModelId: () => string | null | undefined;
}

export interface FileErrorData {
	generallyUnsupported: File[];
	modalityUnsupported: File[];
	modalityReasons: Record<string, string>;
	supportedTypes: string[];
}

export function useChatScreenFileUpload(options: UseChatScreenFileUploadOptions) {
	let uploadedFiles = $state<ChatUploadedFile[]>([]);
	let showFileErrorDialog = $state(false);
	let fileErrorData = $state<FileErrorData>({
		generallyUnsupported: [],
		modalityUnsupported: [],
		modalityReasons: {},
		supportedTypes: []
	});

	async function processFiles(files: File[]) {
		const generallySupported: File[] = [];
		const generallyUnsupported: File[] = [];

		for (const file of files) {
			if (isFileTypeSupported(file.name, file.type)) {
				generallySupported.push(file);
			} else {
				generallyUnsupported.push(file);
			}
		}

		const { supportedFiles, unsupportedFiles, modalityReasons } = filterFilesByModalities(
			generallySupported,
			options.capabilities()
		);

		const allUnsupportedFiles = [...generallyUnsupported, ...unsupportedFiles];

		if (allUnsupportedFiles.length > 0) {
			const supportedTypes: string[] = ['text files', 'PDFs'];
			const caps = options.capabilities();
			if (caps.hasVision) supportedTypes.push('images');
			if (caps.hasAudio) supportedTypes.push('audio files');
			if (caps.hasVideo) supportedTypes.push('video files');

			fileErrorData = {
				generallyUnsupported,
				modalityUnsupported: unsupportedFiles,
				modalityReasons,
				supportedTypes
			};
			showFileErrorDialog = true;
		}

		if (supportedFiles.length > 0) {
			const processed = await processFilesToChatUploaded(
				supportedFiles,
				options.activeModelId() ?? undefined
			);
			uploadedFiles = [...uploadedFiles, ...processed];
		}
	}

	function handleFileUpload(files: File[]) {
		return processFiles(files);
	}

	function handleFileRemove(fileId: string) {
		uploadedFiles = uploadedFiles.filter((f) => f.id !== fileId);
	}

	return {
		get uploadedFiles() {
			return uploadedFiles;
		},
		set uploadedFiles(value) {
			uploadedFiles = value;
		},
		get showFileErrorDialog() {
			return showFileErrorDialog;
		},
		set showFileErrorDialog(value) {
			showFileErrorDialog = value;
		},
		fileErrorData,
		handleFileUpload,
		handleFileRemove
	};
}
