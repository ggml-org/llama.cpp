<script lang="ts">
	import ChatForm from '$lib/components/app/chat/ChatForm/ChatForm.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		initialMessage?: string;
		isLoading?: boolean;
		onFileRemove?: (fileId: string) => void;
		onFileUpload?: (files: File[]) => void;
		onSend?: (message: string, files?: ChatUploadedFile[]) => Promise<boolean>;
		onStop?: () => void;
		onSystemPromptAdd?: (draft: { message: string; files: ChatUploadedFile[] }) => void;
		showHelperText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
	}

	let {
		class: className,
		disabled = false,
		initialMessage = '',
		isLoading = false,
		onFileRemove,
		onFileUpload,
		onSend,
		onStop,
		onSystemPromptAdd,
		showHelperText = true,
		uploadedFiles = $bindable([])
	}: Props = $props();
</script>

<div class="relative mx-auto max-w-[48rem]">
	<ChatForm
		class={className}
		{disabled}
		{initialMessage}
		{isLoading}
		{onFileRemove}
		{onFileUpload}
		{onSend}
		{onStop}
		{onSystemPromptAdd}
		{showHelperText}
		bind:uploadedFiles
	/>
</div>
