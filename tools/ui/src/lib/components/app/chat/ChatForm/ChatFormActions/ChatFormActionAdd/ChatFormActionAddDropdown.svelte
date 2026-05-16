<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { buttonVariants } from '$lib/components/ui/button';
	import { cn } from '$lib/components/ui/utils';
	import {
		ATTACHMENT_FILE_ITEMS,
		ATTACHMENT_EXTRA_ITEMS,
		ATTACHMENT_MCP_ITEMS,
		ATTACHMENT_TOOLTIP_TEXT,
		TOOLTIP_DELAY_DURATION
	} from '$lib/constants';
	import { AttachmentMenuItemId } from '$lib/enums';
	import {
		ChatFormActionAddToolsSubmenu,
		ChatFormActionAddMcpServersSubmenu
	} from '$lib/components/app';

	import { useAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVisionModality?: boolean;
		hasMcpPromptsSupport?: boolean;
		hasMcpResourcesSupport?: boolean;
		onFileUpload?: () => void;
		onSystemPromptClick?: () => void;
		onMcpPromptClick?: () => void;
		onMcpSettingsClick?: () => void;
		onMcpResourcesClick?: () => void;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		hasVisionModality = false,
		hasMcpPromptsSupport = false,
		hasMcpResourcesSupport = false,
		onFileUpload,
		onSystemPromptClick,
		onMcpPromptClick,
		onMcpSettingsClick,
		onMcpResourcesClick
	}: Props = $props();

	let dropdownOpen = $state(false);

	function handleMcpSettingsClick() {
		dropdownOpen = false;
		onMcpSettingsClick?.();
	}

	const attachmentMenu = useAttachmentMenu(
		() => ({
			hasVisionModality,
			hasAudioModality,
			hasMcpPromptsSupport,
			hasMcpResourcesSupport
		}),
		() => ({ onFileUpload, onSystemPromptClick, onMcpPromptClick, onMcpResourcesClick }),
		() => {
			dropdownOpen = false;
		}
	);
</script>

<div class="flex items-center gap-1 {className}">
	<DropdownMenu.Root bind:open={dropdownOpen}>
		<Tooltip.Root>
			<Tooltip.Trigger>
				{#snippet child({ props })}
					<DropdownMenu.Trigger
						{...props}
						class={cn(
							buttonVariants({ variant: 'secondary' }),
							'file-upload-button h-8 w-8 cursor-pointer rounded-full p-0'
						)}
						{disabled}
					>
						<span class="sr-only">{ATTACHMENT_TOOLTIP_TEXT}</span>

						<Plus class="h-4 w-4" />
					</DropdownMenu.Trigger>
				{/snippet}
			</Tooltip.Trigger>

			<Tooltip.Content>
				<p>{ATTACHMENT_TOOLTIP_TEXT}</p>
			</Tooltip.Content>
		</Tooltip.Root>

		<DropdownMenu.Content align="start" class="w-48">
			{#each ATTACHMENT_FILE_ITEMS as item (item.id)}
				{@const enabled = attachmentMenu.isItemEnabled(item.enabledWhen)}
				{#if enabled}
					<DropdownMenu.Item
						class="{item.class ?? ''} flex cursor-pointer items-center gap-2"
						onclick={() => attachmentMenu.callbacks[item.action]()}
					>
						<item.icon class="h-4 w-4" />

						<span>{item.label}</span>
					</DropdownMenu.Item>
				{:else if item.disabledTooltip}
					<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
						<Tooltip.Trigger tabindex={-1}>
							{#snippet child({ props })}
								<div {...props} class="cursor-default">
									<DropdownMenu.Item class="{item.class ?? ''} flex items-center gap-2" disabled>
										<item.icon class="h-4 w-4" />

										<span>{item.label}</span>
									</DropdownMenu.Item>
								</div>
							{/snippet}
						</Tooltip.Trigger>

						<Tooltip.Content side="right">
							<p>{item.disabledTooltip}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}
			{/each}

			{#if !attachmentMenu.isItemEnabled('hasVisionModality')}
				<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
					<Tooltip.Trigger>
						{#snippet child({ props })}
							<DropdownMenu.Item
								{...props}
								class="flex cursor-pointer items-center gap-2"
								onclick={attachmentMenu.callbacks.onFileUpload}
							>
								{@const pdfItem = ATTACHMENT_FILE_ITEMS.find(
									(i) => i.id === AttachmentMenuItemId.PDF
								)}
								{#if pdfItem}
									<pdfItem.icon class="h-4 w-4" />

									<span>{pdfItem.label}</span>
								{/if}
							</DropdownMenu.Item>
						{/snippet}
					</Tooltip.Trigger>

					<Tooltip.Content side="right">
						<p>PDFs will be converted to text. Image-based PDFs may not work properly.</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			<DropdownMenu.Separator />

			{#each ATTACHMENT_EXTRA_ITEMS as item (item.id)}
				{#if item.id === AttachmentMenuItemId.SYSTEM_MESSAGE}
					<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
						<Tooltip.Trigger>
							{#snippet child({ props })}
								<DropdownMenu.Item
									{...props}
									class="flex cursor-pointer items-center gap-2"
									onclick={() => attachmentMenu.callbacks[item.action]()}
								>
									<item.icon class="h-4 w-4" />

									<span>{item.label}</span>
								</DropdownMenu.Item>
							{/snippet}
						</Tooltip.Trigger>

						<Tooltip.Content side="right">
							<p>{attachmentMenu.getSystemMessageTooltip()}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}
			{/each}

			<ChatFormActionAddToolsSubmenu />

			<ChatFormActionAddMcpServersSubmenu onMcpSettingsClick={handleMcpSettingsClick} />

			{#each ATTACHMENT_MCP_ITEMS as item (item.id)}
				{#if attachmentMenu.isItemVisible(item.visibleWhen)}
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={() => attachmentMenu.callbacks[item.action]()}
					>
						<item.icon class="h-4 w-4" />

						<span>{item.label}</span>
					</DropdownMenu.Item>
				{/if}
			{/each}
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
