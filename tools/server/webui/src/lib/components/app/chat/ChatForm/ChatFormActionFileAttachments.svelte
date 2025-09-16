<script lang="ts">
	import { Paperclip, Image, FileText, File, Volume2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants/tooltip-config';
	import { FileTypeCategory } from '$lib/enums/files';
	import { supportsAudio, supportsVision } from '$lib/stores/server.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		onFileUpload?: (fileType?: FileTypeCategory) => void;
	}

	let { class: className = '', disabled = false, onFileUpload }: Props = $props();

	const fileUploadTooltipText = $derived.by(() => {
		return !supportsVision()
			? 'Text files and PDFs supported. Images, audio, and video require vision models.'
			: 'Attach files';
	});

	function handleFileUpload(fileType?: FileTypeCategory) {
		onFileUpload?.(fileType);
	}
</script>

<div class="flex items-center gap-1 {className}">
	<DropdownMenu.Root>
		<DropdownMenu.Trigger name="Attach files">
			<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
				<Tooltip.Trigger>
					<Button
						class="file-upload-button h-8 w-8 rounded-full bg-transparent p-0 text-muted-foreground hover:bg-foreground/10 hover:text-foreground"
						{disabled}
						type="button"
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
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="images-button flex cursor-pointer items-center gap-2"
						disabled={!supportsVision()}
						onclick={() => handleFileUpload(FileTypeCategory.IMAGE)}
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
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="audio-button flex cursor-pointer items-center gap-2"
						disabled={!supportsAudio()}
						onclick={() => handleFileUpload(FileTypeCategory.AUDIO)}
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
				class="flex cursor-pointer items-center gap-2"
				onclick={() => handleFileUpload(FileTypeCategory.TEXT)}
			>
				<FileText class="h-4 w-4" />

				<span>Text Files</span>
			</DropdownMenu.Item>

			<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={() => handleFileUpload(FileTypeCategory.PDF)}
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
