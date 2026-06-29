<script lang="ts">
	import type { Snippet } from 'svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Sheet from '$lib/components/ui/sheet';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { File, MessageSquare, Zap, FolderOpen } from '@lucide/svelte';
	import { Switch } from '$lib/components/ui/switch';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants';
	import { ATTACHMENT_FILE_ITEMS } from '$lib/constants/attachment-menu';
	import { useAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';
	import { useToolsPanel } from '$lib/hooks/use-tools-panel.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { McpLogo } from '$lib/components/app';
	import { PencilRuler, ChevronDown, ChevronRight } from '@lucide/svelte';
	import { HealthCheckStatus } from '$lib/enums';
	import { AttachmentAction } from '$lib/enums/attachment.enums';

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
		onMcpResourcesClick?: () => void;
		trigger: Snippet<[{ disabled: boolean; onclick?: () => void }]>;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		hasVisionModality = false,
		hasVideoModality = false,
		hasMcpPromptsSupport = false,
		hasMcpResourcesSupport = false,
		onFileUpload,
		onSystemPromptClick,
		onMcpPromptClick,
		onMcpResourcesClick,
		trigger
	}: Props = $props();

	let sheetOpen = $state(false);
	let filesExpanded = $state(true);
	let toolsExpanded = $state(false);
	let mcpExpanded = $state(false);

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
			sheetOpen = false;
		}
	);

	const toolsPanel = useToolsPanel();

	const sheetItemClass =
		'flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm transition-colors hover:bg-accent active:bg-accent disabled:cursor-not-allowed disabled:opacity-50';

	const sheetItemRowClass =
		'flex w-full items-center justify-between gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent';

	function getEnabledMcpServers() {
		return mcpStore.getServersSorted().filter((s) => s.enabled);
	}
</script>

<div class="flex items-center gap-1 {className}">
	<Sheet.Root bind:open={sheetOpen}>
		{@render trigger({ disabled, onclick: () => (sheetOpen = true) })}

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
							<ChevronDown class="h-4 w-4 shrink-0" />
						{:else}
							<ChevronRight class="h-4 w-4 shrink-0" />
						{/if}

						<File class="h-4 w-4 shrink-0" />

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
						</div>
					</Collapsible.Content>
				</Collapsible.Root>

				<Collapsible.Root open={mcpExpanded} onOpenChange={(open) => (mcpExpanded = open)}>
					<Collapsible.Trigger class={sheetItemClass}>
						{#if mcpExpanded}
							<ChevronDown class="h-4 w-4 shrink-0" />
						{:else}
							<ChevronRight class="h-4 w-4 shrink-0" />
						{/if}

						<McpLogo class="inline h-4 w-4 shrink-0" />

						<span class="flex-1">MCP Servers</span>

						<span class="text-xs text-muted-foreground">
							{getEnabledMcpServers().length} server{getEnabledMcpServers().length !== 1 ? 's' : ''}
						</span>
					</Collapsible.Trigger>

					<Collapsible.Content>
						<div class="flex flex-col gap-0.5 pl-4">
							{#each getEnabledMcpServers() as server (server.id)}
								{@const healthState = mcpStore.getHealthCheckState(server.id)}
								{@const hasError = healthState.status === HealthCheckStatus.ERROR}
								{@const displayName = mcpStore.getServerLabel(server)}
								{@const faviconUrl = mcpStore.getServerFavicon(server.id)}
								{@const isEnabled = conversationsStore.isMcpServerEnabledForChat(server.id)}

								<button
									type="button"
									class={sheetItemRowClass}
									onclick={() => !hasError && conversationsStore.toggleMcpServerForChat(server.id)}
									disabled={hasError}
								>
									<div class="flex min-w-0 flex-1 items-center gap-2">
										{#if faviconUrl}
											<img
												src={faviconUrl}
												alt=""
												class="h-4 w-4 shrink-0 rounded-sm"
												onerror={(e) => {
													(e.currentTarget as HTMLImageElement).style.display = 'none';
												}}
											/>
										{/if}

										<span class="min-w-0 truncate text-sm">{displayName}</span>
									</div>

									{#if hasError}
										<span
											class="shrink-0 rounded bg-destructive/15 px-1.5 py-0.5 text-xs text-destructive"
										>
											Error
										</span>
									{:else}
										<Switch
											checked={isEnabled}
											onCheckedChange={() => conversationsStore.toggleMcpServerForChat(server.id)}
										/>
									{/if}
								</button>
							{/each}

							{#if getEnabledMcpServers().length === 0}
								<div class="px-3 py-2 text-center text-sm text-muted-foreground">
									No MCP servers configured
								</div>
							{/if}
						</div>
					</Collapsible.Content>
				</Collapsible.Root>

				{#if toolsPanel.totalToolCount > 0}
					<Collapsible.Root open={toolsExpanded} onOpenChange={(open) => (toolsExpanded = open)}>
						<Collapsible.Trigger class={sheetItemClass}>
							{#if toolsExpanded}
								<ChevronDown class="h-4 w-4 shrink-0" />
							{:else}
								<ChevronRight class="h-4 w-4 shrink-0" />
							{/if}

							<PencilRuler class="inline h-4 w-4 shrink-0" />

							<span class="flex-1">Tools</span>

							<span class="text-xs text-muted-foreground">
								{toolsPanel.totalToolCount} tool{toolsPanel.totalToolCount !== 1 ? 's' : ''}
							</span>
						</Collapsible.Trigger>

						<Collapsible.Content>
							<div class="flex flex-col gap-0.5 pl-4">
								{#each toolsPanel.activeGroups as group (group.label)}
									{@const checked = toolsPanel.isGroupChecked(group)}
									{@const enabledCount = toolsPanel.getEnabledToolCount(group)}
									{@const favicon = toolsPanel.getFavicon(group)}

									<button
										type="button"
										class={sheetItemRowClass}
										onclick={() => toolsPanel.toggleGroupByLabel(group.label)}
									>
										{#if favicon}
											<img
												src={favicon}
												alt=""
												class="h-4 w-4 shrink-0 rounded-sm"
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
											class="h-4 w-4 shrink-0"
											onclick={(e) => e.stopPropagation()}
											onCheckedChange={() => toolsPanel.toggleGroupByLabel(group.label)}
										/>
									</button>
								{/each}
							</div>
						</Collapsible.Content>
					</Collapsible.Root>
				{/if}

				<button
					type="button"
					class={sheetItemClass}
					onclick={() => attachmentMenu.callbacks[AttachmentAction.SYSTEM_PROMPT_CLICK]()}
				>
					<MessageSquare class="h-4 w-4 shrink-0" />

					<span>System Message</span>
				</button>

				{#if hasMcpPromptsSupport}
					<button
						type="button"
						class={sheetItemClass}
						onclick={() => attachmentMenu.callbacks[AttachmentAction.MCP_PROMPT_CLICK]()}
					>
						<Zap class="h-4 w-4 shrink-0" />

						<span>MCP Prompt</span>
					</button>
				{/if}

				{#if hasMcpResourcesSupport}
					<button
						type="button"
						class={sheetItemClass}
						onclick={() => attachmentMenu.callbacks[AttachmentAction.MCP_RESOURCES_CLICK]()}
					>
						<FolderOpen class="h-4 w-4 shrink-0" />

						<span>MCP Resources</span>
					</button>
				{/if}
			</div>
		</Sheet.Content>
	</Sheet.Root>
</div>
