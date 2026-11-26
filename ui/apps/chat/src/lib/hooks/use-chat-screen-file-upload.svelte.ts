/**
 * File upload lifecycle for the ChatScreen form.
 *
 * Owns the queue of processed `ChatUploadedFile`, the rejection-by-capability
 * dialog state, and the dual-layer validation pipeline (general format +
 * model modality). The caller provides the active model's capabilities and ID
 * as reactive getters so validation tracks the model in real time.
 */

import { FILE_URI_PREFIX } from '#lib/constants/index.ts';
import { ReadMediaService } from '#lib/services/read-media.service.ts';
import { processFilesToChatUploaded } from '#lib/utils/browser-only.ts';
import {
	encodeFileLinkPath,
	filterFilesByModalities,
	getMediaMimeTypeByExtension,
	isFileTypeSupported,
	isFileTypeSupportedByModel
} from '#lib/utils/index.ts';
import { toast } from 'svelte-sonner';

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
		modalityReasons: {},
		modalityUnsupported: [],
		supportedTypes: []
	});
	// the message as last reported, to tell whether a mention is still in it
	let messageText = '';

	const isMentioned = (path: string) =>
		messageText.includes(FILE_URI_PREFIX + encodeFileLinkPath(path));

	async function processFiles(files: File[], mentionPath?: string) {
		const generallySupported: File[] = [];
		const generallyUnsupported: File[] = [];

		for (const file of files) {
			if (isFileTypeSupported(file.name, file.type)) {
				generallySupported.push(file);
			} else {
				generallyUnsupported.push(file);
			}
		}

		const { modalityReasons, supportedFiles, unsupportedFiles } = filterFilesByModalities(
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
				modalityReasons,
				modalityUnsupported: unsupportedFiles,
				supportedTypes
			};
			showFileErrorDialog = true;
		}

		if (supportedFiles.length > 0) {
			const processed = await processFilesToChatUploaded(
				supportedFiles,
				options.activeModelId() ?? undefined
			);

			uploadedFiles = [
				...uploadedFiles,
				...processed.map((file) => (mentionPath ? { ...file, mentionPath } : file))
			];
		}
	}

	function handleFileUpload(files: File[]) {
		return processFiles(files);
	}

	/**
	 * A file picked with an `@` mention: an image, audio clip or video the model can take
	 * is read from the server and attached like an uploaded one, so the model gets the file
	 * itself and not only its path. Anything else stays a mention.
	 */
	async function handleFileMention(path: string) {
		const mimeType = getMediaMimeTypeByExtension(path);

		if (!mimeType || !isFileTypeSupportedByModel(path, mimeType, options.capabilities())) return;

		// mentioned twice, attached once
		if (uploadedFiles.some((file) => file.mentionPath === path)) return;

		try {
			const file = await ReadMediaService.readFile(path, mimeType);

			// the mention may be gone again by the time the file is read
			if (isMentioned(path)) await processFiles([file], path);
		} catch (error) {
			const reason = error instanceof Error ? error.message : String(error);

			toast.error(`Could not attach the mentioned file: ${reason}`);
		}
	}

	/** Told the message on each change: an attachment leaves with the mention that added it. */
	function syncFileMentions(text: string) {
		messageText = text;

		const kept = uploadedFiles.filter((file) => !file.mentionPath || isMentioned(file.mentionPath));

		if (kept.length !== uploadedFiles.length) uploadedFiles = kept;
	}

	function handleFileRemove(fileId: string) {
		uploadedFiles = uploadedFiles.filter((f) => f.id !== fileId);
	}

	return {
		get fileErrorData() {
			return fileErrorData;
		},
		handleFileMention,
		handleFileRemove,
		handleFileUpload,
		get showFileErrorDialog() {
			return showFileErrorDialog;
		},
		set showFileErrorDialog(value) {
			showFileErrorDialog = value;
		},
		syncFileMentions,
		get uploadedFiles() {
			return uploadedFiles;
		},
		set uploadedFiles(value) {
			uploadedFiles = value;
		}
	};
}
