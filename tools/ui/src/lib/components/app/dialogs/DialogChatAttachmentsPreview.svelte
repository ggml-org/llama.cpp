<script lang="ts">
	import { X } from '@lucide/svelte';
	import { ChatAttachmentsPreview } from '$lib/components/app';
	import * as DialogUI from '$lib/components/ui/dialog';
	import { KeyboardKey } from '$lib/enums';
	import { Dialog } from 'bits-ui';

	interface Props {
		open: boolean;
		uploadedFiles?: ChatUploadedFile[];
		attachments?: DatabaseMessageExtra[];
		activeModelId?: string;
		previewFocusIndex?: number;
	}

	let {
		activeModelId,
		attachments = [],
		open = $bindable(false),
		previewFocusIndex = 0,
		uploadedFiles = []
	}: Props = $props();

	function handleClose() {
		open = false;
	}

	$effect(() => {
		if (!open) return;

		function handleKeyDown(event: KeyboardEvent) {
			const target = event.target as HTMLElement;

			if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

			switch (event.key) {
				case KeyboardKey.ARROW_LEFT:
					event.preventDefault();
					event.stopPropagation();

					document.dispatchEvent(new CustomEvent('chat-attachments-nav', { detail: -1 }));

					break;
				case KeyboardKey.ARROW_RIGHT:
					event.preventDefault();
					event.stopPropagation();

					document.dispatchEvent(new CustomEvent('chat-attachments-nav', { detail: 1 }));

					break;
				case KeyboardKey.SPACE:
					event.preventDefault();
					event.stopPropagation();

					document.dispatchEvent(new CustomEvent('chat-attachments-nav', { detail: 1 }));

					break;
			}
		}

		document.addEventListener('keydown', handleKeyDown);

		return () => document.removeEventListener('keydown', handleKeyDown);
	});
</script>

<Dialog.Root bind:open>
	<Dialog.Portal>
		<DialogUI.Overlay class="bg-black/85" style="z-index: 1000" />

		<Dialog.Content class="fixed inset-0 z-[1000] flex flex-col bg-transparent outline-none">
			<Dialog.Close
				class="absolute top-4 right-4 z-10 cursor-pointer text-white hover:text-gray-400"
				onclick={handleClose}
				aria-label="Close"
			>
				<X class="size-4" />
			</Dialog.Close>

			<ChatAttachmentsPreview
				{uploadedFiles}
				{attachments}
				{activeModelId}
				{previewFocusIndex}
				class="min-h-0 flex-1"
			/>
		</Dialog.Content>
	</Dialog.Portal>
</Dialog.Root>
