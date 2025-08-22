<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Paperclip, Image, FileText, File, Volume2 } from '@lucide/svelte';
	import { supportsAudio, supportsVision } from '$lib/stores/server.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants/tooltip-config';

	interface Props {
		disabled?: boolean;
		onFileUpload?: (fileType?: 'image' | 'audio' | 'file' | 'pdf') => void;
		class?: string;
	}

	let {
		disabled = false,
		onFileUpload,
		class: className = ''
	}: Props = $props();

	const fileUploadTooltipText = !supportsVision() 
		? 'Text files and PDFs supported. Images, audio, and video require vision models.'
		: 'Attach files';

	function handleFileUpload(fileType?: 'image' | 'audio' | 'file' | 'pdf') {
		onFileUpload?.(fileType);
	}
</script>

<div class="flex items-center gap-1 {className}">
	<DropdownMenu.Root>
		<DropdownMenu.Trigger name="Attach files">
			<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
				<Tooltip.Trigger>
					<Button
						type="button"
						class="file-upload-button text-muted-foreground bg-transparent hover:bg-foreground/10 hover:text-foreground h-8 w-8 rounded-full p-0"
						{disabled}
					>
						<span class="sr-only">Attach files</span>

						<Paperclip class="h-4 w-4" />
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>{fileUploadTooltipText}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		</DropdownMenu.Trigger>

		<DropdownMenu.Content align="start" class="w-48">
			<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
				<Tooltip.Trigger class="w-full" >
					<DropdownMenu.Item 
						class="images-button flex items-center gap-2 cursor-pointer" 
						disabled={!supportsVision()}
						onclick={() => handleFileUpload('image')}
					>
						<Image class="h-4 w-4" />

						<span>Images</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				{#if !supportsVision()}
					<Tooltip.Content>
						<p>Images require vision models to be processed</p>
					</Tooltip.Content>
				{/if}
			</Tooltip.Root>

			<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
				<Tooltip.Trigger class="w-full" >

				<DropdownMenu.Item 
					class="audio-button flex items-center gap-2 cursor-pointer"
					disabled={!supportsAudio()}
					onclick={() => handleFileUpload('audio')}
				>
					<Volume2 class="h-4 w-4" />

					<span>Audio Files</span>
				</DropdownMenu.Item>
			</Tooltip.Trigger>

			{#if !supportsAudio()}
				<Tooltip.Content>
					<p>Audio files require audio models to be processed</p>
				</Tooltip.Content>
			{/if}
			</Tooltip.Root>
			
			<DropdownMenu.Item 
				class="flex items-center gap-2 cursor-pointer" 
				onclick={() => handleFileUpload('file')}
			>
				<FileText class="h-4 w-4" />
				<span>Text Files</span>
			</DropdownMenu.Item>

			<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item 
						class="flex items-center gap-2 cursor-pointer" 
						onclick={() => handleFileUpload('pdf')}
					>
						<File class="h-4 w-4" />

						<span>PDF Files</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				{#if !supportsVision()}
					<Tooltip.Content>
						<p>PDFs will be converted to text. Image-based PDFs may not work properly.</p>
					</Tooltip.Content>
				{/if}
			</Tooltip.Root>
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
