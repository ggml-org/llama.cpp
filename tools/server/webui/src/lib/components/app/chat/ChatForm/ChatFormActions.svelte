<script lang="ts">
	import { Square, ArrowUp } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import ChatFormActionFileAttachments from './ChatFormActionFileAttachments.svelte';
	import ChatFormActionRecord from './ChatFormActionRecord.svelte';
	import type { FileTypeCategory } from '$lib/enums/files';

	interface Props {
		canSend?: boolean;
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		isRecording?: boolean;
		onFileUpload?: (fileType?: FileTypeCategory) => void;
		onMicClick?: () => void;
		onStop?: () => void;
	}

	let {
		canSend = false,
		class: className = '',
		disabled = false,
		isLoading = false,
		isRecording = false,
		onFileUpload,
		onMicClick,
		onStop
	}: Props = $props();
</script>

<div class="flex items-center justify-between gap-1 {className}">
	<ChatFormActionFileAttachments {disabled} {onFileUpload} />

	<div class="flex gap-2">
		{#if isLoading}
			<Button
				type="button"
				onclick={onStop}
				class="h-8 w-8 bg-transparent p-0 hover:bg-destructive/20"
			>
				<span class="sr-only">Stop</span>
				<Square class="h-8 w-8 fill-destructive stroke-destructive" />
			</Button>
		{:else}
			<ChatFormActionRecord {disabled} {isLoading} {isRecording} {onMicClick} />

			<Button
				type="submit"
				disabled={!canSend || disabled || isLoading}
				class="h-8 w-8 rounded-full p-0"
			>
				<span class="sr-only">Send</span>
				<ArrowUp class="h-12 w-12" />
			</Button>
		{/if}
	</div>
</div>
