<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Square, Paperclip, Mic, ArrowUp } from '@lucide/svelte';
	import { supportsAudio, supportsVision } from '$lib/stores/server.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		disabled?: boolean;
		isLoading?: boolean;
		canSend?: boolean;
		onFileUpload?: () => void;
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
	<div class="flex items-center gap-1">
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button
					type="button"
					class="text-muted-foreground bg-transparent hover:bg-foreground/10 hover:text-foreground h-8 w-8 rounded-full p-0 {!supportsVision()
						? 'opacity-50 cursor-not-allowed'
						: ''}"
					disabled={disabled || isLoading || !supportsVision()}
					onclick={onFileUpload}
				>
					<span class="sr-only">Attach files</span>
					<Paperclip class="h-4 w-4" />
				</Button>
			</Tooltip.Trigger>
			{#if !supportsVision()}
				<Tooltip.Content>
					<p>Current model does not support vision</p>
				</Tooltip.Content>
			{/if}
		</Tooltip.Root>
	</div>

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
			<Tooltip.Root>
				<Tooltip.Trigger>
					<Button
						type="button"
						class="h-8 w-8 rounded-full p-0 {isRecording
							? 'bg-red-500 hover:bg-red-600 text-white animate-pulse'
							: 'text-muted-foreground bg-transparent hover:bg-foreground/10 hover:text-foreground'} {!supportsAudio()
							? 'opacity-50 cursor-not-allowed'
							: ''}"
						disabled={disabled || isLoading || !supportsAudio()}
						onclick={onMicClick}
					>
						<span class="sr-only">{isRecording ? 'Stop recording' : 'Start recording'}</span>
						<Mic class="h-4 w-4" />
					</Button>
				</Tooltip.Trigger>

				{#if !supportsAudio()}
					<Tooltip.Content>
						<p>Current model does not support audio</p>
					</Tooltip.Content>
				{/if}
			</Tooltip.Root>

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
