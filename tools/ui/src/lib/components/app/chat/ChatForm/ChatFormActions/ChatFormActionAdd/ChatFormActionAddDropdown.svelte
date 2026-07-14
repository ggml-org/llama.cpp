<script lang="ts">
	import { Plus, File, MessageSquare, Zap, FolderOpen } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { buttonVariants } from '$lib/components/ui/button';
	import { cn } from '$lib/components/ui/utils';
	import {
		ATTACHMENT_FILE_ITEMS,
		ATTACHMENT_TOOLTIP_TEXT,
		TOOLTIP_DELAY_DURATION
	} from '$lib/constants';
	import {
		ChatFormActionAddToolsSubmenu,
		ChatFormActionAddMcpServersSubmenu,
		ChatFormActionAddReasoningSubmenu
	} from '$lib/components/app';
	import { useAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		hasVideoModality?: boolean;
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
		hasVideoModality = false,
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
			hasVideoModality,
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

		<DropdownMenu.Content align="start" class="w-52">
			<ChatFormActionAddReasoningSubmenu />

			<DropdownMenu.Separator />

			<DropdownMenu.Sub>
				<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
					<File class="h-4 w-4" />

					<span>Add files</span>
				</DropdownMenu.SubTrigger>

				<DropdownMenu.SubContent class="w-48">
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
											<DropdownMenu.Item
												class="{item.class ?? ''} flex items-center gap-2"
												disabled
											>
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
				</DropdownMenu.SubContent>
			</DropdownMenu.Sub>

			<DropdownMenu.Item
				class="flex cursor-pointer items-center gap-2"
				onclick={onSystemPromptClick}
			>
				<MessageSquare class="h-4 w-4" />

				<span>System Message</span>
			</DropdownMenu.Item>

			<ChatFormActionAddToolsSubmenu />

			<ChatFormActionAddMcpServersSubmenu onMcpSettingsClick={handleMcpSettingsClick} />

			{#if hasMcpPromptsSupport}
				<DropdownMenu.Separator />

				<DropdownMenu.Item
					class="flex cursor-pointer items-center gap-2"
					onclick={onMcpPromptClick}
				>
					<Zap class="h-4 w-4" />

					<span>MCP Prompt</span>
				</DropdownMenu.Item>
			{/if}

			{#if hasMcpResourcesSupport}
				<DropdownMenu.Item
					class="flex cursor-pointer items-center gap-2"
					onclick={onMcpResourcesClick}
				>
					<FolderOpen class="h-4 w-4" />

					<span>MCP Resources</span>
				</DropdownMenu.Item>
			{/if}
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
