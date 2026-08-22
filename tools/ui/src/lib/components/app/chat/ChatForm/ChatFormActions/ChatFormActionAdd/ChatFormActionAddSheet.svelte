<script lang="ts">
	import { File, MessageSquare } from '@lucide/svelte';
	import { ChevronDown, ChevronRight, PencilRuler } from '@lucide/svelte';
	import { McpLogo } from '$lib/components/app';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as Sheet from '$lib/components/ui/sheet';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import {
		ATTACHMENT_FILE_ITEMS,
		ICON_CLASS_DEFAULT,
		TOOLTIP_DELAY_DURATION
	} from '$lib/constants';
	import { getChatFormActionsContext } from '$lib/contexts';
	import { AttachmentAction } from '$lib/enums/attachment.enums';
	import { useAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';
	import { useToolsPanel } from '$lib/hooks/use-tools-panel.svelte';
	import type { ToolGroup } from '$lib/types';
	import type { Snippet } from 'svelte';

	interface Props {
		class?: string;
		trigger: Snippet<[{ disabled: boolean; onclick?: () => void }]>;
	}

	let { class: className = '', trigger }: Props = $props();

	const chatFormActions = getChatFormActionsContext();

	let sheetOpen = $state(false);
	let filesExpanded = $state(true);
	let toolsExpanded = $state(false);

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
			sheetOpen = false;
		}
	);

	const toolsPanel = useToolsPanel();

	const sheetItemClass =
		'flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm transition-colors hover:bg-accent active:bg-accent disabled:cursor-not-allowed disabled:opacity-50';

	const sheetItemRowClass =
		'flex w-full items-center justify-between gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent';
</script>

<div class="flex items-center gap-1 {className}">
	<Sheet.Root bind:open={sheetOpen}>
		{@render trigger({ disabled: chatFormActions.disabled, onclick: () => (sheetOpen = true) })}

		<Sheet.Content side="bottom" class="max-h-[85vh] gap-0 overflow-y-auto">
			<Sheet.Header>
				<Sheet.Title>Add to chat</Sheet.Title>

				<Sheet.Description class="sr-only">
					Add files, system prompt or configure MCP servers
				</Sheet.Description>
			</Sheet.Header>

			<div class="flex flex-col gap-1 px-1.5 pb-2">
				<Collapsible.Root open={filesExpanded} onOpenChange={(open) => (filesExpanded = open)}>
					<Collapsible.Trigger class={sheetItemClass}>
						{#if filesExpanded}
							<ChevronDown class="{ICON_CLASS_DEFAULT} shrink-0" />
						{:else}
							<ChevronRight class="{ICON_CLASS_DEFAULT} shrink-0" />
						{/if}

						<File class="{ICON_CLASS_DEFAULT} shrink-0" />

						<span class="flex-1">Add files</span>
					</Collapsible.Trigger>

					<Collapsible.Content>
						<div class="flex flex-col gap-0.5 pl-4">
							{#each ATTACHMENT_FILE_ITEMS as item (item.id)}
								{@const enabled = attachmentMenu.isItemEnabled(item.enabledWhen)}
								{#if enabled}
									<button
										type="button"
										class={sheetItemClass}
										onclick={() => attachmentMenu.callbacks[item.action]()}
									>
										<item.icon class="{ICON_CLASS_DEFAULT} shrink-0" />

										<span>{item.label}</span>
									</button>
								{:else if item.disabledTooltip}
									<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
										<Tooltip.Trigger>
											<button type="button" class={sheetItemClass} disabled>
												<item.icon class="{ICON_CLASS_DEFAULT} shrink-0" />

												<span>{item.label}</span>
											</button>
										</Tooltip.Trigger>

										<Tooltip.Content side="right">
											<p>{item.disabledTooltip}</p>
										</Tooltip.Content>
									</Tooltip.Root>
								{/if}
							{/each}
						</div>
					</Collapsible.Content>
				</Collapsible.Root>

				<button
					type="button"
					class={sheetItemClass}
					onclick={() => attachmentMenu.callbacks[AttachmentAction.SYSTEM_PROMPT_CLICK]()}
				>
					<MessageSquare class="{ICON_CLASS_DEFAULT} shrink-0" />

					<span>System Message</span>
				</button>

				{#if toolsPanel.totalToolCount > 0}
					<Collapsible.Root open={toolsExpanded} onOpenChange={(open) => (toolsExpanded = open)}>
						<Collapsible.Trigger class={sheetItemClass}>
							{#if toolsExpanded}
								<ChevronDown class="{ICON_CLASS_DEFAULT} shrink-0" />
							{:else}
								<ChevronRight class="{ICON_CLASS_DEFAULT} shrink-0" />
							{/if}

							<PencilRuler class="inline {ICON_CLASS_DEFAULT} shrink-0" />

							<span class="flex-1">Tools</span>

							<span class="text-xs text-muted-foreground">
								{toolsPanel.totalToolCount} tool{toolsPanel.totalToolCount !== 1 ? 's' : ''}
							</span>
						</Collapsible.Trigger>

						<Collapsible.Content>
							<div class="flex flex-col gap-0.5 pl-4">
								{#each toolsPanel.categoryGroups as group (group.key)}
									{@render sheetGroupRow(group)}
								{/each}

								{#each toolsPanel.mcpGroups as group (group.key)}
									{@render sheetGroupRow(group)}
								{/each}
							</div>
						</Collapsible.Content>
					</Collapsible.Root>
				{/if}

				<button
					type="button"
					class={sheetItemClass}
					onclick={() => chatFormActions.onMcpSettingsClick?.()}
				>
					<McpLogo class="inline {ICON_CLASS_DEFAULT} shrink-0" />

					<span>MCP Servers</span>
				</button>
			</div>
		</Sheet.Content>
	</Sheet.Root>
</div>

{#snippet sheetGroupRow(group: ToolGroup)}
	{@const checked = toolsPanel.isGroupChecked(group)}
	{@const enabledCount = toolsPanel.getEnabledToolCount(group)}
	{@const favicon = toolsPanel.getFavicon(group)}
	{@const groupDisabled = toolsPanel.isGroupDisabled(group)}

	<button
		type="button"
		class="{sheetItemRowClass} {groupDisabled ? 'opacity-50' : ''}"
		onclick={() => toolsPanel.toggleGroupByKey(group.key)}
	>
		{#if favicon}
			<img
				src={favicon}
				alt=""
				class="{ICON_CLASS_DEFAULT} shrink-0 rounded-sm"
				onerror={(e) => {
					(e.currentTarget as HTMLImageElement).style.display = 'none';
				}}
			/>
		{/if}

		<span class="min-w-0 flex-1 truncate text-sm font-medium">{group.label}</span>

		<span class="shrink-0 text-xs text-muted-foreground">
			{enabledCount}/{group.tools.length}
		</span>

		<Checkbox
			{checked}
			class="{ICON_CLASS_DEFAULT} shrink-0"
			onclick={(e) => e.stopPropagation()}
			onCheckedChange={() => toolsPanel.toggleGroupByKey(group.key)}
		/>
	</button>
{/snippet}
