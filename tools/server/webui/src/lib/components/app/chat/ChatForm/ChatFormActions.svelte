<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Square, ArrowUp } from '@lucide/svelte';
	import ChatFormActionFileAttachments from './ChatFormActionFileAttachments.svelte';
	import ChatFormActionRecord from './ChatFormActionRecord.svelte';

	interface Props {
		disabled?: boolean;
		isLoading?: boolean;
		canSend?: boolean;
		onFileUpload?: (fileType?: 'image' | 'audio' | 'file' | 'pdf') => void;
		onStop?: () => void;
		onMicClick?: () => void;
		isRecording?: boolean;
		class?: string;
	}

	let {
		disabled = false,
		isLoading = false,
		canSend = false,
		onFileUpload,
		onStop,
		onMicClick,
		isRecording = false,
		class: className = ''
	}: Props = $props();
</script>

<div class="flex items-center justify-between gap-1 {className}">
	<ChatFormActionFileAttachments 
		disabled={disabled}
		{onFileUpload}
	/>

	<div class="flex gap-2">
		{#if isLoading}
			<Button
				type="button"
				onclick={onStop}
				class="p-0 h-8 w-8 bg-transparent hover:bg-destructive/20"
			>
				<span class="sr-only">Stop</span>
				<Square class="h-8 w-8 fill-destructive stroke-destructive" />
			</Button>
		{:else}
			<ChatFormActionRecord 
				{disabled}
				{isLoading}
				{isRecording}
				{onMicClick}
			/>

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
