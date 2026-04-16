<script lang="ts">
	import { page } from '$app/state';
	import { Plus, PencilRuler, ChevronDown, ChevronRight, Loader2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { Switch } from '$lib/components/ui/switch';
	import * as Sheet from '$lib/components/ui/sheet';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants';
	import {
		ATTACHMENT_FILE_ITEMS,
		ATTACHMENT_EXTRA_ITEMS,
		ATTACHMENT_MCP_ITEMS,
		ATTACHMENT_TOOLTIP_TEXT
	} from '$lib/constants/attachment-menu';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { TruncatedText } from '$lib/components/app';
	import { useToolsPanel } from '$lib/hooks/use-tools-panel.svelte';

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

	let sheetOpen = $state(false);
	let hoveredGroup = $state<string | null>(null);

	const {
		expandedGroups,
		groups,
		totalToolCount,
		getGroupCheckedState,
		getFavicon,
		isGroupDisabled,
		toggleGroupExpanded,
		handleOpen
	} = useToolsPanel();

	const callbacks = $derived({
		onFileUpload: () => closeAndCall(onFileUpload),
		onSystemPromptClick: () => closeAndCall(onSystemPromptClick),
		onMcpPromptClick: () => closeAndCall(onMcpPromptClick),
		onMcpResourcesClick: () => closeAndCall(onMcpResourcesClick)
	});

	const modalityFlags = $derived({
		hasVisionModality,
		hasAudioModality,
		hasMcpPromptsSupport,
		hasMcpResourcesSupport
	});

	const sheetItemClass =
		'flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm transition-colors hover:bg-accent active:bg-accent disabled:cursor-not-allowed disabled:opacity-50';

	function isItemEnabled(enabledWhen: string | undefined): boolean {
		if (!enabledWhen || enabledWhen === 'always') return true;
		return !!modalityFlags[enabledWhen as keyof typeof modalityFlags];
	}

	function isItemVisible(visibleWhen: string | undefined): boolean {
		if (!visibleWhen) return true;
		return !!modalityFlags[visibleWhen as keyof typeof modalityFlags];
	}

	function getSystemMessageTooltip(): string {
		return !page.params.id
			? 'Add custom system message for a new conversation'
			: 'Inject custom system message at the beginning of the conversation';
	}

	async function toggleServerForChat(serverId: string) {
		await conversationsStore.toggleMcpServerForChat(serverId);
	}

	function closeAndCall(fn?: () => void) {
		sheetOpen = false;
		fn?.();
	}
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
					{@const enabled = isItemEnabled(item.enabledWhen)}
					{#if enabled}
						<button type="button" class={sheetItemClass} onclick={() => callbacks[item.action]()}>
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

				{#if !isItemEnabled('hasVisionModality')}
					{@const pdfItem = ATTACHMENT_FILE_ITEMS.find((i) => i.id === 'pdf')}
					{#if pdfItem}
						<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
							<Tooltip.Trigger>
								<button
									type="button"
									class={sheetItemClass}
									onclick={() => callbacks[pdfItem.action]()}
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
					{#if item.id === 'system-message'}
						<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
							<Tooltip.Trigger>
								<button
									type="button"
									class={sheetItemClass}
									onclick={() => callbacks[item.action]()}
								>
									<item.icon class="h-4 w-4 shrink-0" />

									<span>{item.label}</span>
								</button>
							</Tooltip.Trigger>

							<Tooltip.Content side="right">
								<p>{getSystemMessageTooltip()}</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{/if}
				{/each}

				<div class="my-2 border-t"></div>

				<button type="button" class={sheetItemClass} onclick={() => handleOpen()}>
					<PencilRuler class="h-4 w-4 shrink-0" />

					<span>Tools</span>
				</button>

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
								onOpenChange={() => toggleGroupExpanded(group.label)}
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
										<Tooltip.Root>
											<Tooltip.Trigger>
												<Switch
													checked={false}
													onclick={(e: MouseEvent) => e.stopPropagation()}
													onCheckedChange={() =>
														group.serverId && toggleServerForChat(group.serverId)}
													class="mr-2 shrink-0"
												/>
											</Tooltip.Trigger>
											<Tooltip.Content side="left">
												<p>Enable {group.label}</p>
											</Tooltip.Content>
										</Tooltip.Root>
									{:else}
										<Tooltip.Root>
											<Tooltip.Trigger>
												<Checkbox
													{checked}
													{indeterminate}
													disabled={groupDisabled}
													onCheckedChange={() => toolsStore.toggleGroup(group)}
													class="mr-2 h-4 w-4 shrink-0 {groupDisabled ? 'opacity-40' : ''}"
												/>
											</Tooltip.Trigger>
											<Tooltip.Content side="right">
												<p>
													{checked ? 'Disable' : 'Enable'}
													{group.tools.length} tool{group.tools.length !== 1 ? 's' : ''}
												</p>
											</Tooltip.Content>
										</Tooltip.Root>
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
												onclick={() => !groupDisabled && toolsStore.toggleTool(tool.function.name)}
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

												<TruncatedText
													text={tool.function.name}
													class="min-w-0 flex-1 truncate"
													showTooltip={true}
												/>
											</button>
										{/each}
									</div>
								</Collapsible.Content>
							</Collapsible.Root>
						{/each}
					</div>
				{/if}

				{#each ATTACHMENT_MCP_ITEMS as item (item.id)}
					{#if isItemVisible(item.visibleWhen)}
						<button type="button" class={sheetItemClass} onclick={() => callbacks[item.action]()}>
							<item.icon class="h-4 w-4 shrink-0" />

							<span>{item.label}</span>
						</button>
					{/if}
				{/each}
			</div>
		</Sheet.Content>
	</Sheet.Root>
</div>
