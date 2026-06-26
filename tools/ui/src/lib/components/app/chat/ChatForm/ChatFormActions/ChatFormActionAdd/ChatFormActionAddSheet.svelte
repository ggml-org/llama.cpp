<script lang="ts">
	import type { Snippet } from 'svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Sheet from '$lib/components/ui/sheet';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import {
		File,
		MessageSquare,
		Plus,
		FolderOpen,
		PencilRuler,
		ChevronDown,
		ChevronRight
	} from '@lucide/svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Badge } from '$lib/components/ui/badge';
	import { TOOLTIP_DELAY_DURATION, PROMPT_CONTENT_SEPARATOR } from '$lib/constants';
	import { ATTACHMENT_FILE_ITEMS } from '$lib/constants/attachment-menu';
	import { useAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';
	import { useToolsPanel } from '$lib/hooks/use-tools-panel.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { promptsStore } from '$lib/stores/prompts.svelte';
	import { McpLogo } from '$lib/components/app';
	import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
	import { ContentPartType } from '$lib/enums';
	import { AttachmentAction } from '$lib/enums/attachment.enums';
	import type { MCPPromptInfo, PromptMessage } from '$lib/types';
	import { buildMcpPromptId } from '$lib/utils';
	import { SvelteMap } from 'svelte/reactivity';

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
		onSystemPromptWithContent?: (content: string, promptId?: string, title?: string) => void;
		onMcpPromptClick?: (prompt?: MCPPromptInfo) => void;
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
		onSystemPromptWithContent,
		onMcpPromptClick,
		onMcpResourcesClick,
		trigger
	}: Props = $props();

	let sheetOpen = $state(false);
	let filesExpanded = $state(false);
	let toolsExpanded = $state(false);
	let promptsExpanded = $state(false);
	let searchQuery = $state('');

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

	let prompts = $derived(promptsStore.getPrompts());

	// Single search field shared by both lists below. Matching is
	// case-insensitive across title/name, content/description, and (for MCP)
	// the server label.
	let filteredLibraryPrompts = $derived.by(() => {
		const q = searchQuery.trim().toLowerCase();
		if (!q) return prompts;
		return prompts.filter(
			(p) => p.title.toLowerCase().includes(q) || p.content.toLowerCase().includes(q)
		);
	});

	// Trigger MCP connection establishment as the sheet opens. Prompts are
	// pulled eagerly inside `MCPService.connect` and surfaced synchronously via
	// `mcpStore.allPrompts`, so the listing renders without a fetch on open.
	$effect(() => {
		if (sheetOpen) {
			void mcpStore.ensureInitialized(conversationsStore.getAllMcpServerOverrides());
		}
	});

	let serverSettingsMap = $derived.by(() => {
		const servers = mcpStore.getServers();
		const map = new SvelteMap<string, ReturnType<typeof mcpStore.getServers>[number]>();
		for (const server of servers) {
			map.set(server.id, server);
		}
		return map;
	});

	let filteredMcpPrompts = $derived.by(() => {
		const sortedServers = mcpStore.getServersSorted();
		const serverOrderMap = new Map(sortedServers.map((server, index) => [server.id, index]));

		const entries = mcpStore.allPrompts
			.slice()
			.sort((a, b) => {
				const orderA = serverOrderMap.get(a.serverName) ?? Number.MAX_SAFE_INTEGER;
				const orderB = serverOrderMap.get(b.serverName) ?? Number.MAX_SAFE_INTEGER;
				return orderA - orderB;
			})
			.map((prompt) => {
				const server = serverSettingsMap.get(prompt.serverName);
				return {
					prompt,
					server,
					serverLabel: server ? mcpStore.getServerLabel(server) : prompt.serverName,
					faviconUrl: server ? mcpStore.getServerFavicon(server.id) : null
				};
			});

		const q = searchQuery.trim().toLowerCase();
		if (!q) return entries;
		return entries.filter((entry) => {
			const name = (entry.prompt.title || entry.prompt.name || '').toLowerCase();
			const description = (entry.prompt.description || '').toLowerCase();
			const server = entry.serverLabel.toLowerCase();
			return name.includes(q) || description.includes(q) || server.includes(q);
		});
	});

	function handlePromptClick(promptId: string) {
		const prompt = promptsStore.getPrompt(promptId);
		if (prompt && onSystemPromptWithContent) {
			sheetOpen = false;
			onSystemPromptWithContent(prompt.content, promptId, prompt.title);
		}
	}

	function handleAddPromptClick() {
		attachmentMenu.callbacks[AttachmentAction.SYSTEM_PROMPT_CLICK]();
	}

	async function handleMcpPromptClick(entry: (typeof filteredMcpPrompts)[number]) {
		sheetOpen = false;

		// Prompts with arguments need a picker detour (for the argument form);
		// prompts without arguments post directly so the conversation gets the
		// system prompt instantly, like the library prompt branch above.
		if (entry.prompt.arguments?.length) {
			onMcpPromptClick?.(entry.prompt);
			return;
		}

		void executeAndPostAsSystemPrompt(entry.prompt);
	}

	function mcpPromptId(prompt: MCPPromptInfo): string {
		return `mcp:${prompt.serverName}:${prompt.name}`;
	}

	async function executeAndPostAsSystemPrompt(prompt: MCPPromptInfo) {
		if (!onSystemPromptWithContent) return;

		try {
			const result = await mcpStore.getPrompt(prompt.serverName, prompt.name, {});

			if (!result?.messages) return;

			const text = result.messages
				.map((msg: PromptMessage) => {
					if (typeof msg.content === 'string') return msg.content;
					if (msg.content.type === ContentPartType.TEXT) return msg.content.text;
					return '';
				})
				.filter(Boolean)
				.join(PROMPT_CONTENT_SEPARATOR);

			if (!text) return;

			onSystemPromptWithContent(text, mcpPromptId(prompt), prompt.title || prompt.name);
		} catch (error) {
			console.warn('[ChatFormActionAddSheet] Failed to execute MCP prompt:', error);
		}
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

				<Collapsible.Root open={promptsExpanded} onOpenChange={(open) => (promptsExpanded = open)}>
					<Collapsible.Trigger class={sheetItemClass}>
						{#if promptsExpanded}
							<ChevronDown class="h-4 w-4 shrink-0" />
						{:else}
							<ChevronRight class="h-4 w-4 shrink-0" />
						{/if}

						<MessageSquare class="h-4 w-4 shrink-0" />

						<span class="flex-1">System message</span>
					</Collapsible.Trigger>

					<Collapsible.Content>
						<div class="flex flex-col gap-0.5 pl-4">
							{#if prompts.length > 0 || mcpStore.allPrompts.length > 0}
								<div class="mb-1.5 mt-1">
									<SearchInput
										bind:value={searchQuery}
										variant="sm"
										placeholder="Search by name, content, or server..."
									/>
								</div>
							{/if}

							<button
								type="button"
								class="{sheetItemClass} mt-2.5 mb-2"
								onclick={handleAddPromptClick}
							>
								<Plus class="h-4 w-4 shrink-0" />

								<span>Write new</span>
							</button>

							{#if filteredLibraryPrompts.length > 0}
								<div class="mt-1.5 mb-1 px-3 text-xs font-medium text-muted-foreground">
									Your Prompts
								</div>

								{#each filteredLibraryPrompts as prompt (prompt.id)}
									<button
										type="button"
										class="flex w-full cursor-pointer flex-col items-start gap-0.5 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent active:bg-accent"
										onclick={() => handlePromptClick(prompt.id)}
									>
										<span class="line-clamp-1 text-sm font-medium">{prompt.title}</span>
										<span
											class="line-clamp-3 w-full text-left text-xs leading-5 text-muted-foreground"
										>
											{prompt.content}
										</span>
									</button>
								{/each}
							{/if}

							{#if filteredMcpPrompts.length > 0}
								<div class="mt-1.5 mb-1 px-3 text-xs font-medium text-muted-foreground">
									MCP Prompts
								</div>

								{#each filteredMcpPrompts as entry (entry)}
									<button
										type="button"
										class="flex w-full cursor-pointer flex-col items-start gap-0.5 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent active:bg-accent"
										onclick={() => handleMcpPromptClick(entry)}
									>
										<div class="flex w-full items-center gap-1.5">
											{#if entry.faviconUrl}
												<img
													src={entry.faviconUrl}
													alt=""
													class="h-3 w-3 shrink-0 rounded-sm"
													onerror={(e) => {
														(e.currentTarget as HTMLImageElement).style.display = 'none';
													}}
												/>
											{:else}
												<McpLogo class="h-3 w-3 shrink-0" />
											{/if}

											<span class="truncate text-xs text-muted-foreground">{entry.serverLabel}</span
											>

											{#if entry.prompt.arguments?.length}
												<Badge variant="secondary" class="ml-auto shrink-0 text-[10px]">
													{entry.prompt.arguments.length} arg{entry.prompt.arguments.length > 1
														? 's'
														: ''}
												</Badge>
											{/if}
										</div>

										<span class="line-clamp-1 text-sm font-medium">
											{entry.prompt.title || entry.prompt.name}
										</span>

										{#if entry.prompt.description}
											<span
												class="line-clamp-2 w-full text-left text-xs leading-5 text-muted-foreground"
											>
												{entry.prompt.description}
											</span>
										{/if}
									</button>
								{/each}
							{/if}

							{#if searchQuery.trim() !== '' && (prompts.length > 0 || mcpStore.allPrompts.length > 0) && filteredLibraryPrompts.length === 0 && filteredMcpPrompts.length === 0}
								<div class="px-3 py-2 text-xs text-muted-foreground">No matches</div>
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
