<script lang="ts">
	import { Paperclip } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { FILE_TYPE_ICONS } from '$lib/constants/icons';
	import { t } from '$lib/i18n';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVisionModality?: boolean;
		onFileUpload?: () => void;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		hasVisionModality = false,
		onFileUpload
	}: Props = $props();

	const fileUploadTooltipText = $derived.by(() => {
		return !hasVisionModality
			? t('chat.form.attachments.tooltip_text_only')
			: t('chat.form.attachments.tooltip_default');
	});
</script>

<div class="flex items-center gap-1 {className}">
	<DropdownMenu.Root>
		<DropdownMenu.Trigger name={t('chat.form.attachments.trigger_label')} {disabled}>
			<Tooltip.Root>
				<Tooltip.Trigger>
					<Button
						class="file-upload-button h-8 w-8 rounded-full bg-transparent p-0 text-muted-foreground hover:bg-foreground/10 hover:text-foreground"
						{disabled}
						type="button"
					>
						<span class="sr-only">{t('chat.form.attachments.trigger_label')}</span>

						<Paperclip class="h-4 w-4" />
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>{fileUploadTooltipText}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		</DropdownMenu.Trigger>

		<DropdownMenu.Content align="start" class="w-48">
			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="images-button flex cursor-pointer items-center gap-2"
						disabled={!hasVisionModality}
						onclick={() => onFileUpload?.()}
					>
						<FILE_TYPE_ICONS.image class="h-4 w-4" />

						<span>{t('chat.form.attachments.images')}</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				{#if !hasVisionModality}
					<Tooltip.Content>
						<p>{t('chat.form.attachments.images_requires_vision')}</p>
					</Tooltip.Content>
				{/if}
			</Tooltip.Root>

			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="audio-button flex cursor-pointer items-center gap-2"
						disabled={!hasAudioModality}
						onclick={() => onFileUpload?.()}
					>
						<FILE_TYPE_ICONS.audio class="h-4 w-4" />

						<span>{t('chat.form.attachments.audio')}</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				{#if !hasAudioModality}
					<Tooltip.Content>
						<p>{t('chat.form.attachments.audio_requires_model')}</p>
					</Tooltip.Content>
				{/if}
			</Tooltip.Root>

			<DropdownMenu.Item
				class="flex cursor-pointer items-center gap-2"
				onclick={() => onFileUpload?.()}
			>
				<FILE_TYPE_ICONS.text class="h-4 w-4" />

				<span>{t('chat.form.attachments.text')}</span>
			</DropdownMenu.Item>

			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={() => onFileUpload?.()}
					>
						<FILE_TYPE_ICONS.pdf class="h-4 w-4" />

						<span>{t('chat.form.attachments.pdf')}</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				{#if !hasVisionModality}
					<Tooltip.Content>
						<p>{t('chat.form.attachments.pdf_text_warning')}</p>
					</Tooltip.Content>
				{/if}
			</Tooltip.Root>
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
