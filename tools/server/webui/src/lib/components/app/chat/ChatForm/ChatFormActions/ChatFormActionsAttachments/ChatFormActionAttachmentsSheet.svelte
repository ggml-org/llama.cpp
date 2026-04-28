<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Sheet from '$lib/components/ui/sheet';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants';
	import {
		ATTACHMENT_FILE_ITEMS,
		ATTACHMENT_EXTRA_ITEMS,
		ATTACHMENT_MCP_ITEMS,
		ATTACHMENT_TOOLTIP_TEXT
	} from '$lib/constants/attachment-menu';
	import { ChatFormActionToolsSubmenu, ChatFormActionMcpServersSubmenu } from '$lib/components/app';
	import { useAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';
	import { AttachmentMenuItemId } from '$lib/enums';

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

	let sheetOpen = $state(false);

	const attachmentMenu = useAttachmentMenu(
		() => ({ hasVisionModality, hasAudioModality, hasMcpPromptsSupport, hasMcpResourcesSupport }),
		() => ({ onFileUpload, onSystemPromptClick, onMcpPromptClick, onMcpResourcesClick }),
		() => {
			sheetOpen = false;
		}
	);

	function handleMcpSettingsClick() {
		sheetOpen = false;
		onMcpSettingsClick?.();
	}

	const sheetItemClass =
		'flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm transition-colors hover:bg-accent active:bg-accent disabled:cursor-not-allowed disabled:opacity-50';
</script>

<div class="flex items-center gap-1 {className}">
	<Sheet.Root bind:open={sheetOpen}>
		<Button
			class="file-upload-button h-8 w-8 rounded-full p-0"
			{disabled}
			variant="secondary"
			type="button"
			onclick={() => (sheetOpen = true)}
		>
			<span class="sr-only">{ATTACHMENT_TOOLTIP_TEXT}</span>

			<Plus class="h-4 w-4" />
		</Button>

		<Sheet.Content side="bottom" class="max-h-[85vh] gap-0 overflow-y-auto">
			<Sheet.Header>
				<Sheet.Title>Add to chat</Sheet.Title>

				<Sheet.Description class="sr-only">
					Add files, system prompt or configure MCP servers
				</Sheet.Description>
			</Sheet.Header>

			<div class="flex flex-col gap-1 px-1.5 pb-2">
				{#each ATTACHMENT_FILE_ITEMS as item (item.id)}
					{@const enabled = attachmentMenu.isItemEnabled(item.enabledWhen)}
					{#if enabled}
						<button
							type="button"
							class={sheetItemClass}
							onclick={() => attachmentMenu.callbacks[item.action]()}
						>
							<item.icon class="h-4 w-4 shrink-0" />

							<span>{item.label}</span>
						</button>
					{:else if item.disabledTooltip}
						<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
							<Tooltip.Trigger>
								<button type="button" class={sheetItemClass} disabled>
									<item.icon class="h-4 w-4 shrink-0" />

									<span>{item.label}</span>
								</button>
							</Tooltip.Trigger>

							<Tooltip.Content side="right">
								<p>{item.disabledTooltip}</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{/if}
				{/each}

				{#if !attachmentMenu.isItemEnabled('hasVisionModality')}
					{@const pdfItem = ATTACHMENT_FILE_ITEMS.find((i) => i.id === AttachmentMenuItemId.PDF)}
					{#if pdfItem}
						<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
							<Tooltip.Trigger>
								<button
									type="button"
									class={sheetItemClass}
									onclick={() => attachmentMenu.callbacks[pdfItem.action]()}
								>
									<pdfItem.icon class="h-4 w-4 shrink-0" />

									<span>{pdfItem.label}</span>
								</button>
							</Tooltip.Trigger>

							<Tooltip.Content side="right">
								<p>PDFs will be converted to text. Image-based PDFs may not work properly.</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{/if}
				{/if}

				{#each ATTACHMENT_EXTRA_ITEMS as item (item.id)}
					{#if item.id === AttachmentMenuItemId.SYSTEM_MESSAGE}
						<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
							<Tooltip.Trigger>
								<button
									type="button"
									class={sheetItemClass}
									onclick={() => attachmentMenu.callbacks[item.action]()}
								>
									<item.icon class="h-4 w-4 shrink-0" />

									<span>{item.label}</span>
								</button>
							</Tooltip.Trigger>

							<Tooltip.Content side="right">
								<p>{attachmentMenu.getSystemMessageTooltip()}</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{/if}
				{/each}

				<div class="my-2 border-t"></div>

				<ChatFormActionToolsSubmenu />

				<ChatFormActionMcpServersSubmenu onMcpSettingsClick={handleMcpSettingsClick} />

				{#each ATTACHMENT_MCP_ITEMS as item (item.id)}
					{#if attachmentMenu.isItemVisible(item.visibleWhen)}
						<button
							type="button"
							class={sheetItemClass}
							onclick={() => attachmentMenu.callbacks[item.action]()}
						>
							<item.icon class="h-4 w-4 shrink-0" />

							<span>{item.label}</span>
						</button>
					{/if}
				{/each}
			</div>
		</Sheet.Content>
	</Sheet.Root>
</div>
