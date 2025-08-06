<script lang="ts">
	import { ChatAttachmentsList } from '$lib/components/app';
	import { ChatFormActionButtons, ChatFormFileInput, ChatFormHelperText, ChatFormTextarea } from '$lib/components/app';
	import { inputClasses } from '$lib/constants/input-classes';
	import { onMount } from 'svelte';
	import { config } from '$lib/stores/settings.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		onFileRemove?: (fileId: string) => void;
		onFileUpload?: (files: File[]) => void;
		onSend?: (message: string, files?: ChatUploadedFile[]) => Promise<boolean>;
		onStop?: () => void;
		showHelperText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
	}

	let {
		class: className,
		disabled = false,
		isLoading = false,
		onFileRemove,
		onFileUpload,
		onSend,
		onStop,
		showHelperText = true,
		uploadedFiles = $bindable([]),
	}: Props = $props();

	// Get settings
	const currentConfig = $derived(config());
	const pasteLongTextToFileLength = $derived(Number(currentConfig.pasteLongTextToFileLen) || 2500);

	let message = $state('');
	let fileInputRef: ChatFormFileInput | undefined;
	let previousIsLoading = $state(isLoading);
	let textareaRef: ChatFormTextarea | undefined;

	onMount(() => {
		textareaRef?.focus();
	});

	$effect(() => {
		if (previousIsLoading && !isLoading) {
			textareaRef?.focus();
		}

		previousIsLoading = isLoading;
	});

	async function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();

			if (!message.trim() || disabled || isLoading) return;

			const messageToSend = message.trim();
			const filesToSend = [...uploadedFiles];

			message = '';
			uploadedFiles = [];

			textareaRef?.resetHeight();

			const success = await onSend?.(messageToSend, filesToSend);

			if (!success) {
				message = messageToSend;
				uploadedFiles = filesToSend;
			}
		}
	}

	function handleFileSelect(files: File[]) {
		onFileUpload?.(files);
	}

	function handleFileUpload() {
		fileInputRef?.click();
	}

	function handlePaste(event: ClipboardEvent) {
		if (!event.clipboardData) return;

		const files = Array.from(event.clipboardData.items)
			.filter((item) => item.kind === 'file')
			.map((item) => item.getAsFile())
			.filter((file): file is File => file !== null);

		if (files.length > 0) {
			event.preventDefault();
			onFileUpload?.(files);
			return;
		}

		const text = event.clipboardData.getData('text/plain');

		if (
			text.length > 0 &&
			pasteLongTextToFileLength > 0 &&
			text.length > pasteLongTextToFileLength
		) {
			event.preventDefault();

			const textFile = new File([text], 'Pasted', {
				type: 'text/plain'
			});

			onFileUpload?.([textFile]);
		}
	}

	async function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		if (!message.trim() || disabled || isLoading) return;

		const messageToSend = message.trim();
		const filesToSend = [...uploadedFiles];

		message = '';
		uploadedFiles = [];

		textareaRef?.resetHeight();

		const success = await onSend?.(messageToSend, filesToSend);

		if (!success) {
			message = messageToSend;
			uploadedFiles = filesToSend;
		}
	}

	function handleStop() {
		onStop?.();
	}
</script>

<ChatFormFileInput bind:this={fileInputRef} onFileSelect={handleFileSelect} />

<form
	onsubmit={handleSubmit}
	class="{inputClasses} border-radius-bottom-none mx-auto max-w-[48rem] overflow-hidden rounded-3xl backdrop-blur-md {className}"
>
	<ChatAttachmentsList bind:uploadedFiles {onFileRemove} class="mb-3 px-5 pt-5" />

	<div
		class="flex-column relative min-h-[48px] items-center rounded-3xl px-5 py-3 shadow-sm transition-all focus-within:shadow-md"
		onpaste={handlePaste}
	>
		<ChatFormTextarea
			bind:this={textareaRef}
			bind:value={message}
			onKeydown={handleKeydown}
			{disabled}
		/>

		<ChatFormActionButtons
			{disabled}
			{isLoading}
			canSend={message.trim().length > 0}
			onFileUpload={handleFileUpload}
			onStop={handleStop}
		/>
	</div>
</form>

<ChatFormHelperText show={showHelperText} />
