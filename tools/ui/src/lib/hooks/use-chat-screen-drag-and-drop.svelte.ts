/**
 * Drag-and-drop state machine for the ChatScreen.
 *
 * Tracks pointer enter/leave nesting so the overlay stays visible while the
 * cursor traverses child elements, then routes the dropped files either to
 * the active message-edit handler (if a message is being edited) or to the
 * caller's onDrop callback.
 */

import { getAddFilesHandler, isEditing } from '$lib/stores/chat.svelte';

interface UseChatScreenDragAndDropOptions {
	/** Called when the user drops files and no message is being edited. */
	onDrop: (files: File[]) => void;
}

export function useChatScreenDragAndDrop(options: UseChatScreenDragAndDropOptions) {
	let dragCounter = $state(0);
	let isDragOver = $state(false);

	function handleDragEnter(event: DragEvent) {
		event.preventDefault();
		dragCounter++;
		if (event.dataTransfer?.types.includes('Files')) {
			isDragOver = true;
		}
	}

	function handleDragLeave(event: DragEvent) {
		event.preventDefault();
		dragCounter--;
		if (dragCounter === 0) {
			isDragOver = false;
		}
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
	}

	async function handleDrop(event: DragEvent) {
		event.preventDefault();
		isDragOver = false;
		dragCounter = 0;

		if (!event.dataTransfer?.files) return;

		const files = Array.from(event.dataTransfer.files);

		if (isEditing()) {
			const handler = getAddFilesHandler();
			if (handler) {
				handler(files);
				return;
			}
		}

		options.onDrop(files);
	}

	return {
		get isDragOver() {
			return isDragOver;
		},
		dragHandlers: {
			dragenter: handleDragEnter,
			dragleave: handleDragLeave,
			dragover: handleDragOver,
			drop: handleDrop
		}
	};
}
