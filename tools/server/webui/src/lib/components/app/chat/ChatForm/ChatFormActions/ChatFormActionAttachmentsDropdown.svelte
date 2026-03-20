<script lang="ts">
	import { page } from '$app/state';
	import {
		Plus,
		MessageSquare,
		Zap,
		FolderOpen,
		PencilRuler,
		ChevronDown,
		ChevronRight,
		Loader2
	} from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { Switch } from '$lib/components/ui/switch';
	import { FILE_TYPE_ICONS, TOOLTIP_DELAY_DURATION } from '$lib/constants';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { toolsStore, type ToolGroup } from '$lib/stores/tools.svelte';
	import { ToolSource } from '$lib/enums';

	import { SvelteSet } from 'svelte/reactivity';

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
		onMcpResourcesClick
	}: Props = $props();

	let isNewChat = $derived(!page.params.id);

	let systemMessageTooltip = $derived(
		isNewChat
			? 'Add custom system message for a new conversation'
			: 'Inject custom system message at the beginning of the conversation'
	);

	let dropdownOpen = $state(false);

	let expandedGroups = new SvelteSet<string>();
	let groups = $derived(toolsStore.toolGroups);
	let activeGroups = $derived(
		groups.filter(
			(g) =>
				g.source !== ToolSource.MCP ||
				!g.serverId ||
				conversationsStore.isMcpServerEnabledForChat(g.serverId)
		)
	);
	let totalToolCount = $derived(activeGroups.reduce((n, g) => n + g.tools.length, 0));

	function isGroupDisabled(group: ToolGroup): boolean {
		return (
			group.source === ToolSource.MCP &&
			!!group.serverId &&
			!conversationsStore.isMcpServerEnabledForChat(group.serverId)
		);
	}
	let hoveredGroup = $state<string | null>(null);

	const fileUploadTooltipText = 'Add files, system prompt or MCP Servers';

	function getGroupCheckedState(group: (typeof groups)[number]): {
		checked: boolean;
		indeterminate: boolean;
	} {
		return {
			checked: toolsStore.isGroupFullyEnabled(group),
			indeterminate: toolsStore.isGroupPartiallyEnabled(group)
		};
	}

	function getFavicon(group: { source: ToolSource; label: string }): string | null {
		if (group.source !== ToolSource.MCP) return null;

		for (const server of mcpStore.getServersSorted()) {
			if (mcpStore.getServerLabel(server) === group.label) {
				return mcpStore.getServerFavicon(server.id);
			}
		}

		return null;
	}

	function handleToolsSubMenuOpen(open: boolean) {
		if (open) {
			if (toolsStore.builtinTools.length === 0 && !toolsStore.loading) {
				toolsStore.fetchBuiltinTools();
			}
			mcpStore.runHealthChecksForServers(mcpStore.getServersSorted().filter((s) => s.enabled));
		}
	}
	async function toggleServerForChat(serverId: string) {
		await conversationsStore.toggleMcpServerForChat(serverId);
	}

	function handleMcpPromptClick() {
		dropdownOpen = false;
		onMcpPromptClick?.();
	}

	function handleMcpResourcesClick() {
		dropdownOpen = false;
		onMcpResourcesClick?.();
	}
</script>

<div class="flex items-center gap-1 {className}">
	<DropdownMenu.Root bind:open={dropdownOpen}>
		<DropdownMenu.Trigger name="Attach files" {disabled}>
			<Tooltip.Root>
				<Tooltip.Trigger class="w-full">
					<Button
						class="file-upload-button h-8 w-8 rounded-full p-0"
						{disabled}
						variant="secondary"
						type="button"
					>
						<span class="sr-only">{fileUploadTooltipText}</span>

						<Plus class="h-4 w-4" />
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>{fileUploadTooltipText}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		</DropdownMenu.Trigger>

		<DropdownMenu.Content align="start" class="w-48">
			{#if hasVisionModality}
				<DropdownMenu.Item
					class="images-button flex cursor-pointer items-center gap-2"
					onclick={() => onFileUpload?.()}
				>
					<FILE_TYPE_ICONS.image class="h-4 w-4" />

					<span>Images</span>
				</DropdownMenu.Item>
			{:else}
				<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
					<Tooltip.Trigger class="w-full">
						<DropdownMenu.Item
							class="images-button flex cursor-pointer items-center gap-2"
							disabled
						>
							<FILE_TYPE_ICONS.image class="h-4 w-4" />

							<span>Images</span>
						</DropdownMenu.Item>
					</Tooltip.Trigger>

					<Tooltip.Content side="right">
						<p>Image processing requires a vision model</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			{#if hasAudioModality}
				<DropdownMenu.Item
					class="audio-button flex cursor-pointer items-center gap-2"
					onclick={() => onFileUpload?.()}
				>
					<FILE_TYPE_ICONS.audio class="h-4 w-4" />

					<span>Audio Files</span>
				</DropdownMenu.Item>
			{:else}
				<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
					<Tooltip.Trigger class="w-full">
						<DropdownMenu.Item class="audio-button flex cursor-pointer items-center gap-2" disabled>
							<FILE_TYPE_ICONS.audio class="h-4 w-4" />

							<span>Audio Files</span>
						</DropdownMenu.Item>
					</Tooltip.Trigger>

					<Tooltip.Content side="right">
						<p>Audio files processing requires an audio model</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			<DropdownMenu.Item
				class="flex cursor-pointer items-center gap-2"
				onclick={() => onFileUpload?.()}
			>
				<FILE_TYPE_ICONS.text class="h-4 w-4" />

				<span>Text Files</span>
			</DropdownMenu.Item>

			{#if hasVisionModality}
				<DropdownMenu.Item
					class="flex cursor-pointer items-center gap-2"
					onclick={() => onFileUpload?.()}
				>
					<FILE_TYPE_ICONS.pdf class="h-4 w-4" />

					<span>PDF Files</span>
				</DropdownMenu.Item>
			{:else}
				<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
					<Tooltip.Trigger class="w-full">
						<DropdownMenu.Item
							class="flex cursor-pointer items-center gap-2"
							onclick={() => onFileUpload?.()}
						>
							<FILE_TYPE_ICONS.pdf class="h-4 w-4" />

							<span>PDF Files</span>
						</DropdownMenu.Item>
					</Tooltip.Trigger>

					<Tooltip.Content side="right">
						<p>PDFs will be converted to text. Image-based PDFs may not work properly.</p>
					</Tooltip.Content>
				</Tooltip.Root>
			{/if}

			<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
				<Tooltip.Trigger class="w-full">
					<DropdownMenu.Item
						class="flex cursor-pointer items-center gap-2"
						onclick={() => onSystemPromptClick?.()}
					>
						<MessageSquare class="h-4 w-4" />

						<span>System Message</span>
					</DropdownMenu.Item>
				</Tooltip.Trigger>

				<Tooltip.Content side="right">
					<p>{systemMessageTooltip}</p>
				</Tooltip.Content>
			</Tooltip.Root>

			<DropdownMenu.Separator />

			<DropdownMenu.Sub onOpenChange={handleToolsSubMenuOpen}>
				<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
					<PencilRuler class="h-4 w-4" />

					<span>Tools</span>
				</DropdownMenu.SubTrigger>

				<DropdownMenu.SubContent class="w-72 p-0">
					{#if totalToolCount === 0 && groups.length === 0}
						<div class="px-3 py-4 text-center text-sm text-muted-foreground">
							{#if toolsStore.loading}
								<Loader2 class="mx-auto mb-1 h-4 w-4 animate-spin" />
								Loading tools...
							{:else if toolsStore.error}
								Failed to load tools
							{:else}
								No tools available
							{/if}
						</div>
					{:else}
						<div class="max-h-80 overflow-y-auto p-2 pr-1">
							{#each groups as group (group.label)}
								{@const groupDisabled = isGroupDisabled(group)}
								{@const isExpanded = expandedGroups.has(group.label)}
								{@const { checked, indeterminate } = groupDisabled
									? { checked: false, indeterminate: false }
									: getGroupCheckedState(group)}
								{@const favicon = getFavicon(group)}

								<Collapsible.Root
									open={isExpanded}
									onOpenChange={() => {
										if (expandedGroups.has(group.label)) {
											expandedGroups.delete(group.label);
										} else {
											expandedGroups.add(group.label);
										}
									}}
								>
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<div
										class="flex items-center gap-1"
										onmouseenter={() => {
											if (groupDisabled) hoveredGroup = group.label;
										}}
										onmouseleave={() => {
											if (hoveredGroup === group.label) hoveredGroup = null;
										}}
									>
										<Collapsible.Trigger
											class="flex min-w-0 flex-1 items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-muted/50 {groupDisabled
												? 'opacity-40'
												: ''}"
										>
											{#if isExpanded}
												<ChevronDown class="h-3.5 w-3.5 shrink-0" />
											{:else}
												<ChevronRight class="h-3.5 w-3.5 shrink-0" />
											{/if}

											<span class="inline-flex min-w-0 items-center gap-1.5 font-medium">
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

												<span class="truncate">{group.label}</span>
											</span>

											<span class="ml-auto shrink-0 text-xs text-muted-foreground">
												{group.tools.length}
											</span>
										</Collapsible.Trigger>

										{#if groupDisabled && hoveredGroup === group.label && group.serverId}
											<Switch
												checked={false}
												onclick={(e: MouseEvent) => e.stopPropagation()}
												onCheckedChange={() =>
													group.serverId && toggleServerForChat(group.serverId)}
												class="mr-2 shrink-0"
											/>
										{:else}
											<Checkbox
												{checked}
												{indeterminate}
												disabled={groupDisabled}
												onCheckedChange={() => toolsStore.toggleGroup(group)}
												class="mr-2 h-4 w-4 shrink-0 {groupDisabled ? 'opacity-40' : ''}"
											/>
										{/if}
									</div>

									<Collapsible.Content>
										<div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
											{#each group.tools as tool (tool.function.name)}
												<button
													type="button"
													class="flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm transition-colors {groupDisabled
														? 'pointer-events-none opacity-40'
														: 'hover:bg-muted/50'}"
													onclick={() =>
														!groupDisabled && toolsStore.toggleTool(tool.function.name)}
												>
													<Checkbox
														checked={groupDisabled
															? false
															: toolsStore.isToolEnabled(tool.function.name)}
														disabled={groupDisabled}
														onCheckedChange={() =>
															!groupDisabled && toolsStore.toggleTool(tool.function.name)}
														class="h-4 w-4 shrink-0"
													/>

													<span class="min-w-0 flex-1 truncate">
														{tool.function.name}
													</span>
												</button>
											{/each}
										</div>
									</Collapsible.Content>
								</Collapsible.Root>
							{/each}
						</div>

						<!-- <div class="px-3 py-2 text-xs font-medium text-muted-foreground">
							{enabledToolCount}/{totalToolCount} tools enabled
						</div> -->
					{/if}
				</DropdownMenu.SubContent>
			</DropdownMenu.Sub>

			{#if hasMcpPromptsSupport}
				<DropdownMenu.Item
					class="flex cursor-pointer items-center gap-2"
					onclick={handleMcpPromptClick}
				>
					<Zap class="h-4 w-4" />

					<span>MCP Prompt</span>
				</DropdownMenu.Item>
			{/if}

			{#if hasMcpResourcesSupport}
				<DropdownMenu.Item
					class="flex cursor-pointer items-center gap-2"
					onclick={handleMcpResourcesClick}
				>
					<FolderOpen class="h-4 w-4" />

					<span>MCP Resources</span>
				</DropdownMenu.Item>
			{/if}
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
