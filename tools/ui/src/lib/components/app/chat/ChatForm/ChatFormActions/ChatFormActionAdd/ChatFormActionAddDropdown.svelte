<script lang="ts">
	import { File, Image, MessageSquare, Mic, Plus, Video } from '@lucide/svelte';
	import { ChatFormActionAddToolsSubmenu, McpLogo } from '$lib/components/app';
	import { buttonVariants } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';
	import {
		ATTACHMENT_FILE_ITEMS,
		ATTACHMENT_TOOLTIP_TEXT,
		ICON_CLASS_DEFAULT
	} from '$lib/constants';
	import { getChatFormActionsContext } from '$lib/contexts';
	import { AttachmentAction, AttachmentItemEnabledWhen } from '$lib/enums';
	import { useAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';

	interface Props {
		class?: string;
	}

	let { class: className = '' }: Props = $props();

	const chatFormActions = getChatFormActionsContext();

	let dropdownOpen = $state(false);
	// The system message action moves focus to the message editor, so the menu
	// must not restore focus to the trigger on close
	let suppressCloseAutoFocus = false;

	const attachmentMenu = useAttachmentMenu(
		() => ({
			hasAudioModality: chatFormActions.hasAudioModality,
			hasMcpPromptsSupport: chatFormActions.hasMcpPromptsSupport,
			hasMcpResourcesSupport: chatFormActions.hasMcpResourcesSupport,
			hasVideoModality: chatFormActions.hasVideoModality,
			hasVisionModality: chatFormActions.hasVisionModality
		}),
		() => ({
			onFileUpload: chatFormActions.onFileUpload,
			onMcpPromptClick: chatFormActions.onMcpPromptClick,
			onMcpResourcesClick: chatFormActions.onMcpResourcesClick,
			onSystemPromptClick: chatFormActions.onSystemPromptClick
		}),
		() => {
			dropdownOpen = false;
		}
	);

	const FILE_MODALITY_ICONS: Record<string, { icon: typeof Image; label: string }> = {
		[AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY]: { icon: Mic, label: 'Audio' },
		[AttachmentItemEnabledWhen.HAS_VIDEO_MODALITY]: { icon: Video, label: 'Video' },
		[AttachmentItemEnabledWhen.HAS_VISION_MODALITY]: { icon: Image, label: 'Vision' }
	};

	const supportedModalities = $derived.by(() =>
		ATTACHMENT_FILE_ITEMS.filter((item) => attachmentMenu.isItemEnabled(item.enabledWhen))
			.map((item) => FILE_MODALITY_ICONS[item.enabledWhen ?? ''])
			.filter((modality) => modality !== undefined)
	);
</script>

<div class="flex items-center gap-1 {className}">
	<DropdownMenu.Root bind:open={dropdownOpen}>
		<!-- ignoreNonKeyboardFocus prevents the tooltip from flashing when the
		     menu closes and focus returns to the trigger -->
		<Tooltip.Root ignoreNonKeyboardFocus>
			<Tooltip.Trigger>
				{#snippet child({ props })}
					<DropdownMenu.Trigger
						{...props}
						class={cn(
							buttonVariants({ variant: 'secondary' }),
							'file-upload-button h-8 w-8 cursor-pointer rounded-full p-0'
						)}
						disabled={chatFormActions.disabled}
					>
						<span class="sr-only">{ATTACHMENT_TOOLTIP_TEXT}</span>

						<Plus class={ICON_CLASS_DEFAULT} />
					</DropdownMenu.Trigger>
				{/snippet}
			</Tooltip.Trigger>

			<Tooltip.Content>
				<p>{ATTACHMENT_TOOLTIP_TEXT}</p>
			</Tooltip.Content>
		</Tooltip.Root>

		<DropdownMenu.Content
			align="start"
			class="w-52"
			onCloseAutoFocus={(e) => {
				if (suppressCloseAutoFocus) {
					suppressCloseAutoFocus = false;
					e.preventDefault();
				}
			}}
		>
			<DropdownMenu.Item
				class="flex cursor-pointer items-center gap-2"
				onclick={() => attachmentMenu.callbacks[AttachmentAction.FILE_UPLOAD]()}
			>
				<File class={ICON_CLASS_DEFAULT} />

				<span class="flex min-w-0 items-center gap-2">
					<span>Add files</span>

					{#if supportedModalities.length > 0}
						<span class="flex items-center gap-0.75 text-muted-foreground">
							{#each supportedModalities as modality (modality.label)}
								<Tooltip.Root>
									<Tooltip.Trigger>
										<modality.icon class="size-2.75" />
									</Tooltip.Trigger>

									<Tooltip.Content>
										<p>{modality.label}</p>
									</Tooltip.Content>
								</Tooltip.Root>
							{/each}
						</span>
					{/if}
				</span>
			</DropdownMenu.Item>

			<DropdownMenu.Item
				class="flex cursor-pointer items-center gap-2"
				onclick={() => {
					suppressCloseAutoFocus = true;
					chatFormActions.onSystemPromptClick?.();
				}}
			>
				<MessageSquare class={ICON_CLASS_DEFAULT} />

				<span>System Message</span>
			</DropdownMenu.Item>

			<ChatFormActionAddToolsSubmenu />

			<DropdownMenu.Item
				class="flex cursor-pointer items-center gap-2"
				onclick={chatFormActions.onMcpSettingsClick}
			>
				<McpLogo class={ICON_CLASS_DEFAULT} />

				<span>MCP Servers</span>
			</DropdownMenu.Item>
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
