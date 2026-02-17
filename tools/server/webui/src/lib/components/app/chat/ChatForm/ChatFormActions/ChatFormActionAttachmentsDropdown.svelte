<script lang="ts">
	import { page } from '$app/state';
	import { MessageSquare, Plus } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { FILE_TYPE_ICONS } from '$lib/constants/icons';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants/tooltip-config';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVisionModality?: boolean;
		onFileUpload?: () => void;
		onSystemPromptClick?: () => void;
	}

	type AttachmentActionId = 'images' | 'audio' | 'text' | 'pdf' | 'system';

	interface AttachmentAction {
		id: AttachmentActionId;
		label: string;
		disabled?: boolean;
		disabledReason?: string;
		tooltip?: string;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		hasVisionModality = false,
		onFileUpload,
		onSystemPromptClick
	}: Props = $props();

	let isNewChat = $derived(!page.params.id);
	let systemMessageTooltip = $derived(
		isNewChat
			? 'Add custom system message for a new conversation'
			: 'Inject custom system message at the beginning of the conversation'
	);

	let actions = $derived.by<AttachmentAction[]>(() => [
		{
			id: 'images',
			label: 'Images',
			disabled: !hasVisionModality,
			disabledReason: !hasVisionModality
				? 'Images require vision models to be processed'
				: undefined
		},
		{
			id: 'audio',
			label: 'Audio Files',
			disabled: !hasAudioModality,
			disabledReason: !hasAudioModality
				? 'Audio files require audio models to be processed'
				: undefined
		},
		{
			id: 'text',
			label: 'Text Files'
		},
		{
			id: 'pdf',
			label: 'PDF Files',
			tooltip: !hasVisionModality
				? 'PDFs will be converted to text. Image-based PDFs may not work properly.'
				: undefined
		},
		{
			id: 'system',
			label: 'System Message',
			tooltip: systemMessageTooltip
		}
	]);

	function handleActionClick(id: AttachmentActionId) {
		if (id === 'system') {
			onSystemPromptClick?.();
			return;
		}

		onFileUpload?.();
	}

	const triggerTooltipText = 'Add files or system message';
	const itemClass = 'flex cursor-pointer items-center gap-2';
</script>

<div class="flex items-center gap-1 {className}">
	<DropdownMenu.Root>
		<DropdownMenu.Trigger name="Attach files" {disabled}>
			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<Button
						class="file-upload-button h-8 w-8 rounded-full p-0"
						{disabled}
						variant="secondary"
						type="button"
					>
						<span class="sr-only">{triggerTooltipText}</span>

						<Plus class="h-4 w-4" />
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>{triggerTooltipText}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		</DropdownMenu.Trigger>

		<DropdownMenu.Content align="start" class="w-56">
			{#each actions as item (item.id)}
				{@const hasDisabledTooltip = !!item.disabled && !!item.disabledReason}
				{@const hasEnabledTooltip = !item.disabled && !!item.tooltip}

				{#if hasDisabledTooltip}
					<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
						<Tooltip.Trigger class="w-full">
							<DropdownMenu.Item class={itemClass} disabled>
								{#if item.id === 'images'}
									<FILE_TYPE_ICONS.image class="h-4 w-4" />
								{:else if item.id === 'audio'}
									<FILE_TYPE_ICONS.audio class="h-4 w-4" />
								{:else if item.id === 'text'}
									<FILE_TYPE_ICONS.text class="h-4 w-4" />
								{:else if item.id === 'pdf'}
									<FILE_TYPE_ICONS.pdf class="h-4 w-4" />
								{:else}
									<MessageSquare class="h-4 w-4" />
								{/if}

								<span>{item.label}</span>
							</DropdownMenu.Item>
						</Tooltip.Trigger>

						<Tooltip.Content side="right">
							<p>{item.disabledReason}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{:else if hasEnabledTooltip}
					<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
						<Tooltip.Trigger class="w-full">
							<DropdownMenu.Item class={itemClass} onclick={() => handleActionClick(item.id)}>
								{#if item.id === 'images'}
									<FILE_TYPE_ICONS.image class="h-4 w-4" />
								{:else if item.id === 'audio'}
									<FILE_TYPE_ICONS.audio class="h-4 w-4" />
								{:else if item.id === 'text'}
									<FILE_TYPE_ICONS.text class="h-4 w-4" />
								{:else if item.id === 'pdf'}
									<FILE_TYPE_ICONS.pdf class="h-4 w-4" />
								{:else}
									<MessageSquare class="h-4 w-4" />
								{/if}

								<span>{item.label}</span>
							</DropdownMenu.Item>
						</Tooltip.Trigger>

						<Tooltip.Content side="right">
							<p>{item.tooltip}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{:else}
					<DropdownMenu.Item class={itemClass} onclick={() => handleActionClick(item.id)}>
						{#if item.id === 'images'}
							<FILE_TYPE_ICONS.image class="h-4 w-4" />
						{:else if item.id === 'audio'}
							<FILE_TYPE_ICONS.audio class="h-4 w-4" />
						{:else if item.id === 'text'}
							<FILE_TYPE_ICONS.text class="h-4 w-4" />
						{:else if item.id === 'pdf'}
							<FILE_TYPE_ICONS.pdf class="h-4 w-4" />
						{:else}
							<MessageSquare class="h-4 w-4" />
						{/if}

						<span>{item.label}</span>
					</DropdownMenu.Item>
				{/if}
			{/each}
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
