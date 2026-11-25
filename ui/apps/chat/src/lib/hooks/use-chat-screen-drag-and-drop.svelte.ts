/**
 * Drag-and-drop state machine for the ChatScreen.
 *
 * Tracks pointer enter/leave nesting so the overlay stays visible while the
 * cursor traverses child elements. Dropped Llama Chat exports go to the
 * importer; the other files go either to the active message-edit handler (if
 * a message is being edited) or to the caller's onDrop callback. Every drag
 * event stops here, so the app-wide import overlay stays out of the chat.
 */

import { getDropImportContext } from '#lib/contexts/index.ts';
import { chatStore } from '#lib/stores/index.ts';

interface UseChatScreenDragAndDropOptions {
	/** Called when the user drops files and no message is being edited. */
	onDrop: (files: File[]) => void;
}

export function useChatScreenDragAndDrop(options: UseChatScreenDragAndDropOptions) {
	let dragCounter = $state(0);
	let isDragOver = $state(false);

	const dropImport = getDropImportContext();

	function handleDragEnter(event: DragEvent) {
		event.preventDefault();
		event.stopPropagation();
		dragCounter++;

		if (event.dataTransfer?.types.includes('Files')) {
			isDragOver = true;
		}
	}

	function handleDragLeave(event: DragEvent) {
		event.preventDefault();
		event.stopPropagation();
		dragCounter--;

		if (dragCounter === 0) {
			isDragOver = false;
		}
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		event.stopPropagation();
	}

	async function handleDrop(event: DragEvent) {
		event.preventDefault();
		event.stopPropagation();
		isDragOver = false;
		dragCounter = 0;

		if (!event.dataTransfer?.files) return;

		const dropped = Array.from(event.dataTransfer.files);
		const files = dropImport ? await dropImport.importExports(dropped) : dropped;

		if (files.length === 0) return;

		if (chatStore.isEditing()) {
			const handler = chatStore.getAddFilesHandler();

			if (handler) {
				handler(files);

				return;
			}
		}

		options.onDrop(files);
	}

	return {
		dragHandlers: {
			dragenter: handleDragEnter,
			dragleave: handleDragLeave,
			dragover: handleDragOver,
			drop: handleDrop
		},
		get isDragOver() {
			return isDragOver;
		}
	};
}
