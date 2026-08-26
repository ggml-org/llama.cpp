<script lang="ts">
	import { File, FolderOpen, MessageSquare, Zap } from '@lucide/svelte';
	import {
		Check,
		ChevronDown,
		ChevronRight,
		Lightbulb,
		LightbulbOff,
		PencilRuler
	} from '@lucide/svelte';
	import { McpLogo } from '$lib/components/app';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as Sheet from '$lib/components/ui/sheet';
	import { Switch } from '$lib/components/ui/switch';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import {
		ATTACHMENT_FILE_ITEMS,
		ICON_CLASS_DEFAULT,
		TOOLTIP_DELAY_DURATION
	} from '$lib/constants';
	import { getChatFormActionsContext } from '$lib/contexts';
	import { HealthCheckStatus } from '$lib/enums';
	import { AttachmentAction } from '$lib/enums/attachment.enums';
	import { useAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';
	import { useReasoningMenu } from '$lib/hooks/use-reasoning-menu.svelte';
	import { useToolsPanel } from '$lib/hooks/use-tools-panel.svelte';
	import { conversationsStore, mcpStore } from '$lib/stores';
	import type { Snippet } from 'svelte';

	interface Props {
		class?: string;
		trigger: Snippet<[{ disabled: boolean; onclick?: () => void }]>;
	}

	let { class: className = '', trigger }: Props = $props();

	const chatFormActions = getChatFormActionsContext();

	let sheetOpen = $state(false);
	let reasoningExpanded = $state(false);
	let filesExpanded = $state(true);
	let toolsExpanded = $state(false);
	let mcpExpanded = $state(false);

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
	const reasoning = useReasoningMenu();

	const sheetItemClass =
		'flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm transition-colors hover:bg-accent active:bg-accent disabled:cursor-not-allowed disabled:opacity-50';

	const sheetItemRowClass =
		'flex w-full items-center justify-between gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent';

	let mcpServers = $derived(mcpStore.getServers());
</script>

<div class="flex items-center gap-1 {className}">
	<Sheet.Root bind:open={sheetOpen}>
		{@render trigger({ disabled: chatFormActions.disabled, onclick: () => (sheetOpen = true) })}

		<Sheet.Content class="max-h-[85vh] gap-0 overflow-y-auto" side="bottom">
			<Sheet.Header>
				<Sheet.Title>Add to chat</Sheet.Title>

				<Sheet.Description class="sr-only">
					Add files, system prompt or configure MCP servers
				</Sheet.Description>
			</Sheet.Header>

			<div class="flex flex-col gap-1 px-1.5 pb-2">
				{#if reasoning.modelSupportsThinking}
					<Collapsible.Root
						onOpenChange={(open) => (reasoningExpanded = open)}
						open={reasoningExpanded}
					>
						<Collapsible.Trigger class={sheetItemClass}>
							{#if reasoningExpanded}
								<ChevronDown class="{ICON_CLASS_DEFAULT} shrink-0" />
							{:else}
								<ChevronRight class="{ICON_CLASS_DEFAULT} shrink-0" />
							{/if}

							{#if reasoning.thinkingEnabled}
								<Lightbulb class="{ICON_CLASS_DEFAULT} shrink-0 text-amber-400" />
							{:else if reasoning.isOff}
								<LightbulbOff class="{ICON_CLASS_DEFAULT} shrink-0 text-muted-foreground" />
							{:else}
								<Lightbulb class="{ICON_CLASS_DEFAULT} shrink-0 text-muted-foreground" />
							{/if}

							<span class="flex-1">Reasoning</span>

							<span class="text-xs capitalize text-muted-foreground">
								{reasoning.currentEffort}
							</span>
						</Collapsible.Trigger>

						<Collapsible.Content>
							<div class="flex flex-col gap-0.5 pl-4">
								{#each reasoning.levels as level (level.value)}
									{@const tokenLabel = reasoning.tokenLabel(level)}
									<button
										class:bg-accent={reasoning.isSelected(level)}
										class={sheetItemRowClass}
										onclick={() => reasoning.select(level)}
										type="button"
									>
										<div class="flex min-w-0 items-center gap-3">
											{#if reasoning.isSelected(level)}
												<Check class="{ICON_CLASS_DEFAULT} shrink-0 text-foreground" />
											{:else}
												<div class="{ICON_CLASS_DEFAULT} shrink-0"></div>
											{/if}

											<span class="text-sm">{level.label}</span>
										</div>

										{#if tokenLabel}
											<span class="shrink-0 text-[11px] text-muted-foreground opacity-60">
												{tokenLabel}
											</span>
										{/if}
									</button>
								{/each}
							</div>
						</Collapsible.Content>
					</Collapsible.Root>
				{/if}

				<Collapsible.Root onOpenChange={(open) => (filesExpanded = open)} open={filesExpanded}>
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
										class={sheetItemClass}
										onclick={() => attachmentMenu.callbacks[item.action]()}
										type="button"
									>
										<item.icon class="{ICON_CLASS_DEFAULT} shrink-0" />

										<span>{item.label}</span>
									</button>
								{:else if item.disabledTooltip}
									<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
										<Tooltip.Trigger>
											<button class={sheetItemClass} disabled type="button">
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

				<Collapsible.Root onOpenChange={(open) => (mcpExpanded = open)} open={mcpExpanded}>
					<Collapsible.Trigger class={sheetItemClass}>
						{#if mcpExpanded}
							<ChevronDown class="{ICON_CLASS_DEFAULT} shrink-0" />
						{:else}
							<ChevronRight class="{ICON_CLASS_DEFAULT} shrink-0" />
						{/if}

						<McpLogo class="inline {ICON_CLASS_DEFAULT} shrink-0" />

						<span class="flex-1">MCP Servers</span>

						<span class="text-xs text-muted-foreground">
							{mcpServers.length} server{mcpServers.length !== 1 ? 's' : ''}
						</span>
					</Collapsible.Trigger>

					<Collapsible.Content>
						<div class="flex flex-col gap-0.5 pl-4">
							{#each mcpServers as server (server.id)}
								{@const healthState = mcpStore.getHealthCheckState(server.id)}
								{@const hasError = healthState.status === HealthCheckStatus.ERROR}
								{@const displayName = mcpStore.getServerLabel(server)}
								{@const faviconUrl = mcpStore.getServerFavicon(server.id)}
								{@const isEnabled = conversationsStore.preferences.isMcpServerEnabledForChat(
									server.id
								)}

								<button
									class={sheetItemRowClass}
									disabled={hasError}
									onclick={() =>
										!hasError && conversationsStore.preferences.toggleMcpServerForChat(server.id)}
									type="button"
								>
									<div class="flex min-w-0 flex-1 items-center gap-2">
										{#if faviconUrl}
											<img
												alt=""
												class="{ICON_CLASS_DEFAULT} shrink-0 rounded-sm"
												onerror={(e) => {
													(e.currentTarget as HTMLImageElement).style.display = 'none';
												}}
												src={faviconUrl}
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
											onCheckedChange={() =>
												conversationsStore.preferences.toggleMcpServerForChat(server.id)}
										/>
									{/if}
								</button>
							{/each}

							{#if mcpServers.length === 0}
								<div class="px-3 py-2 text-center text-sm text-muted-foreground">
									No MCP servers configured
								</div>
							{/if}
						</div>
					</Collapsible.Content>
				</Collapsible.Root>

				{#if toolsPanel.totalToolCount > 0}
					<Collapsible.Root onOpenChange={(open) => (toolsExpanded = open)} open={toolsExpanded}>
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
								{#each toolsPanel.activeGroups as group (group.key)}
									{@const checked = toolsPanel.isGroupChecked(group)}
									{@const enabledCount = toolsPanel.getEnabledToolCount(group)}
									{@const favicon = toolsPanel.getFavicon(group)}

									<button
										class={sheetItemRowClass}
										onclick={() => toolsPanel.toggleGroupByKey(group.key)}
										type="button"
									>
										{#if favicon}
											<img
												alt=""
												class="{ICON_CLASS_DEFAULT} shrink-0 rounded-sm"
												onerror={(e) => {
													(e.currentTarget as HTMLImageElement).style.display = 'none';
												}}
												src={favicon}
											/>
										{/if}

										<span class="min-w-0 flex-1 truncate text-sm font-medium">{group.label}</span>

										<span class="shrink-0 text-xs text-muted-foreground">
											{enabledCount}/{group.tools.length}
										</span>

										<Checkbox
											{checked}
											class="{ICON_CLASS_DEFAULT} shrink-0"
											onCheckedChange={() => toolsPanel.toggleGroupByKey(group.key)}
											onclick={(e) => e.stopPropagation()}
										/>
									</button>
								{/each}
							</div>
						</Collapsible.Content>
					</Collapsible.Root>
				{/if}

				<button
					class={sheetItemClass}
					onclick={() => attachmentMenu.callbacks[AttachmentAction.SYSTEM_PROMPT_CLICK]()}
					type="button"
				>
					<MessageSquare class="{ICON_CLASS_DEFAULT} shrink-0" />

					<span>System Message</span>
				</button>

				{#if chatFormActions.hasMcpPromptsSupport}
					<button
						class={sheetItemClass}
						onclick={() => attachmentMenu.callbacks[AttachmentAction.MCP_PROMPT_CLICK]()}
						type="button"
					>
						<Zap class="{ICON_CLASS_DEFAULT} shrink-0" />

						<span>MCP Prompt</span>
					</button>
				{/if}

				{#if chatFormActions.hasMcpResourcesSupport}
					<button
						class={sheetItemClass}
						onclick={() => attachmentMenu.callbacks[AttachmentAction.MCP_RESOURCES_CLICK]()}
						type="button"
					>
						<FolderOpen class="{ICON_CLASS_DEFAULT} shrink-0" />

						<span>MCP Resources</span>
					</button>
				{/if}
			</div>
		</Sheet.Content>
	</Sheet.Root>
</div>
