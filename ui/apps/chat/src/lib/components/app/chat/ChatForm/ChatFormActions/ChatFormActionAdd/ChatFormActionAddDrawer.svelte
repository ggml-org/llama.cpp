<script lang="ts">
	import {
		InsetList,
		InsetListRow,
		InsetListSplitRow,
		SheetHeader,
		SheetPages,
		useSheetPages
	} from '@ggml-org/llama-ui/app/sheets';
	import { SHEET_CONTENT_CLASS } from '@ggml-org/llama-ui/constants';
	import * as Drawer from '@ggml-org/llama-ui/ui/drawer';
	import { Switch } from '@ggml-org/llama-ui/ui/switch';
	import {
		Braces,
		Camera,
		File,
		Folder,
		Globe,
		Info,
		Loader2,
		MessageSquare,
		PencilRuler,
		Server
	} from '@lucide/svelte';
	import {
		ATTACHMENT_FILE_ITEMS,
		ATTACHMENT_MENU_TEXT,
		ATTACHMENT_MODALITY_TEXT,
		ICON_CLASS_DEFAULT,
		MODALITY_ICONS,
		SHEET_SHORT_HEIGHT_CLASS
	} from '#lib/constants/index.ts';
	import { getChatFormActionsContext } from '#lib/contexts/index.ts';
	import { AttachmentAction, AttachmentItemEnabledWhen } from '#lib/enums/attachment.enums.ts';
	import { AddSheetPage, ModelModality, ToolSource } from '#lib/enums/index.ts';
	import { useAttachmentMenu } from '#lib/hooks/use-attachment-menu.svelte.ts';
	import { useToolsPanel } from '#lib/hooks/use-tools-panel.svelte.ts';
	import { agenticMode } from '#lib/stores/agentic-mode.svelte.ts';
	import { toolsStore } from '#lib/stores/index.ts';
	import type { ToolGroup } from '#lib/types/index.ts';
	import type { Component, Snippet } from 'svelte';

	interface Props {
		class?: string;
		trigger: Snippet<[{ disabled: boolean; onclick?: () => void }]>;
	}

	let { class: className = '', trigger }: Props = $props();

	const chatFormActions = getChatFormActionsContext();

	/** The sheet's pages: the root menu, the tools list, and one tool group's tools. */
	type Page =
		| { id: AddSheetPage.ROOT }
		| { id: AddSheetPage.TOOLS }
		| { groupKey: string; id: AddSheetPage.TOOL_GROUP };

	let drawerOpen = $state(false);
	const pages = useSheetPages<Page>({ id: AddSheetPage.ROOT });
	const page = $derived(pages.page);

	const TOOL_SOURCE_ICONS: Partial<Record<ToolSource, Component>> = {
		[ToolSource.BROWSER]: Globe,
		[ToolSource.CUSTOM]: Braces,
		[ToolSource.SERVER]: Server
	};

	/** What the camera row's picker takes: a photo. */
	const CAMERA_ACCEPT = 'image/*';

	let cameraInput: HTMLInputElement | undefined = $state();
	let filesInput: HTMLInputElement | undefined = $state();

	const attachmentMenu = useAttachmentMenu(
		() => ({
			hasAudioModality: chatFormActions.hasAudioModality,
			hasVideoModality: chatFormActions.hasVideoModality,
			hasVisionModality: chatFormActions.hasVisionModality
		}),
		() => ({
			onFileUpload: chatFormActions.onFileUpload,
			onSystemPromptClick: chatFormActions.onSystemPromptClick
		}),
		() => {
			drawerOpen = false;
		}
	);

	const FILE_MODALITY_ICONS: Record<string, { icon: Component; label: string }> = {
		[AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY]: {
			icon: MODALITY_ICONS[ModelModality.AUDIO],
			label: ATTACHMENT_MODALITY_TEXT[AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY]
		},
		[AttachmentItemEnabledWhen.HAS_VIDEO_MODALITY]: {
			icon: MODALITY_ICONS[ModelModality.VIDEO],
			label: ATTACHMENT_MODALITY_TEXT[AttachmentItemEnabledWhen.HAS_VIDEO_MODALITY]
		},
		[AttachmentItemEnabledWhen.HAS_VISION_MODALITY]: {
			icon: MODALITY_ICONS[ModelModality.VISION],
			label: ATTACHMENT_MODALITY_TEXT[AttachmentItemEnabledWhen.HAS_VISION_MODALITY]
		}
	};

	const supportedModalities = $derived.by(() =>
		ATTACHMENT_FILE_ITEMS.filter((item) => attachmentMenu.isItemEnabled(item.enabledWhen))
			.map((item) => FILE_MODALITY_ICONS[item.enabledWhen ?? ''])
			.filter((modality) => modality !== undefined)
	);

	// the sheet's own pickers keep it open while the native picker is up; it closes once files arrive
	const canPickFiles = $derived(!!chatFormActions.onFilesSelect);
	// the camera only makes sense for a model that can see what it gives
	const canUseCamera = $derived(chatFormActions.hasVisionModality && canPickFiles);

	const toolsPanel = useToolsPanel();
	const allGroups = $derived([...toolsPanel.categoryGroups, ...toolsPanel.mcpGroups]);
	const enabledToolCount = $derived(
		allGroups.reduce((n, group) => n + toolsPanel.getEnabledToolCount(group), 0)
	);
	const currentGroup = $derived(
		page.id === AddSheetPage.TOOL_GROUP
			? allGroups.find((group) => group.key === page.groupKey)
			: undefined
	);

	const title = $derived.by(() => {
		if (page.id === AddSheetPage.TOOLS) return ATTACHMENT_MENU_TEXT.TOOLS;

		if (page.id === AddSheetPage.TOOL_GROUP)
			return currentGroup?.label ?? ATTACHMENT_MENU_TEXT.TOOLS;

		return ATTACHMENT_MENU_TEXT.ADD_TO_CHAT;
	});

	function open() {
		pages.reset();
		drawerOpen = true;
	}

	function push(next: Page) {
		if (next.id === AddSheetPage.TOOLS) toolsPanel.handleOpen();

		pages.push(next);
	}

	function handleAddFiles() {
		if (canPickFiles) {
			filesInput?.click();
		} else {
			attachmentMenu.callbacks[AttachmentAction.FILE_UPLOAD]();
		}
	}

	function handleFilesSelect(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const files = input.files ? Array.from(input.files) : [];

		// cleared so the same file can be picked again
		input.value = '';

		if (files.length === 0) return;

		drawerOpen = false;
		chatFormActions.onFilesSelect?.(files);
	}
</script>

<!-- hidden pickers: capture opens the camera on phones, the plain input the system file picker -->
<input
	bind:this={cameraInput}
	accept={CAMERA_ACCEPT}
	capture="environment"
	class="hidden"
	onchange={handleFilesSelect}
	type="file"
/>

<input bind:this={filesInput} class="hidden" multiple onchange={handleFilesSelect} type="file" />

<div class="flex items-center gap-1 {className}">
	<Drawer.Root bind:open={drawerOpen}>
		{@render trigger({ disabled: chatFormActions.disabled, onclick: open })}

		<Drawer.Content class="gap-0 {SHEET_SHORT_HEIGHT_CLASS} {SHEET_CONTENT_CLASS}">
			<SheetHeader canGoBack={pages.canGoBack} onBack={pages.pop} {title} />

			<Drawer.Description class="sr-only">{ATTACHMENT_MENU_TEXT.DESCRIPTION}</Drawer.Description>

			<SheetPages
				class="gap-4 px-3 pb-[max(1rem,var(--safe-area-bottom))]"
				keepFirstPageHeight
				pageKey={JSON.stringify(page)}
				{pages}
			>
				{#if page.id === AddSheetPage.ROOT}
					{@render rootPage()}
				{:else if page.id === AddSheetPage.TOOLS}
					{@render toolsPage()}
				{:else if currentGroup}
					{@render groupPage(currentGroup)}
				{/if}
			</SheetPages>
		</Drawer.Content>
	</Drawer.Root>
</div>

{#snippet rootPage()}
	<InsetList>
		{#if canUseCamera}
			<InsetListRow icon={Camera} onclick={() => cameraInput?.click()}>
				{ATTACHMENT_MENU_TEXT.CAMERA}
			</InsetListRow>
		{/if}

		<InsetListRow icon={File} onclick={handleAddFiles}>
			<span class="flex-1">{ATTACHMENT_MENU_TEXT.ADD_FILES}</span>

			{#if supportedModalities.length > 0}
				<span class="flex items-center gap-1 text-muted-foreground">
					{#each supportedModalities as modality (modality.label)}
						<modality.icon aria-label={modality.label} class="size-3" />
					{/each}
				</span>
			{/if}
		</InsetListRow>
	</InsetList>

	<InsetList>
		<InsetListRow
			icon={MessageSquare}
			onclick={() => attachmentMenu.callbacks[AttachmentAction.SYSTEM_PROMPT_CLICK]()}
		>
			{ATTACHMENT_MENU_TEXT.SYSTEM_MESSAGE}
		</InsetListRow>

		{#if chatFormActions.onWorkingDirectoryClick}
			<InsetListRow
				chevron
				icon={Folder}
				onclick={() => {
					drawerOpen = false;
					chatFormActions.onWorkingDirectoryClick?.();
				}}
			>
				<span class="shrink-0">{ATTACHMENT_MENU_TEXT.WORKING_DIRECTORY}</span>

				<span class="min-w-0 flex-1 truncate text-right text-body-sm text-muted-foreground">
					{chatFormActions.workingDirectory ?? ATTACHMENT_MENU_TEXT.NONE}
				</span>
			</InsetListRow>
		{/if}

		<!-- a plain chat has no tools to pick -->
		{#if agenticMode.enabled}
			<InsetListRow chevron icon={PencilRuler} onclick={() => push({ id: AddSheetPage.TOOLS })}>
				<span class="flex-1">{ATTACHMENT_MENU_TEXT.TOOLS}</span>

				{#if toolsPanel.totalToolCount > 0}
					<span class="text-body-sm text-muted-foreground tabular-nums">
						{enabledToolCount}/{toolsPanel.totalToolCount}
					</span>
				{/if}
			</InsetListRow>
		{/if}
	</InsetList>
{/snippet}

{#snippet toolsPage()}
	{#if toolsPanel.totalToolCount === 0}
		<InsetList class="px-4 py-4 text-body-sm text-muted-foreground">
			{#if toolsStore.loading}
				<span class="flex items-center justify-center gap-2">
					<Loader2 class="{ICON_CLASS_DEFAULT} animate-spin" />

					{ATTACHMENT_MENU_TEXT.LOADING_TOOLS}
				</span>
			{:else if toolsStore.error && !toolsStore.isToolsEndpointUnreachable}
				<span class="text-center">{ATTACHMENT_MENU_TEXT.TOOLS_FAILED}</span>
			{:else}
				<span class="flex gap-2">
					<Info class="mt-0.5 {ICON_CLASS_DEFAULT} shrink-0" />

					<span>{toolsPanel.noToolsInfoMessage ?? ATTACHMENT_MENU_TEXT.NO_TOOLS}</span>
				</span>
			{/if}
		</InsetList>
	{:else}
		{#if toolsPanel.categoryGroups.length > 0}
			<InsetList>
				{#each toolsPanel.categoryGroups as group (group.key)}
					{@render groupRow(group)}
				{/each}
			</InsetList>
		{/if}

		{#if toolsPanel.mcpGroups.length > 0}
			<InsetList title={ATTACHMENT_MENU_TEXT.MCP_SERVERS}>
				{#each toolsPanel.mcpGroups as group (group.key)}
					{@render groupRow(group)}
				{/each}
			</InsetList>
		{/if}
	{/if}
{/snippet}

{#snippet groupSwitch(group: ToolGroup, label?: string)}
	<Switch
		aria-label={label}
		checked={toolsPanel.getGroupCheckState(group).checked}
		disabled={toolsPanel.isGroupDisabled(group)}
		onCheckedChange={() => toolsPanel.toggleGroupByKey(group.key)}
		size="lg"
	/>
{/snippet}

{#snippet groupRow(group: ToolGroup)}
	{@const favicon = toolsPanel.getFavicon(group)}
	{@const isChecked = toolsPanel.getGroupCheckState(group).checked}

	<InsetListSplitRow
		dimmed={toolsPanel.isGroupDisabled(group)}
		onclick={() => push({ groupKey: group.key, id: AddSheetPage.TOOL_GROUP })}
	>
		{#snippet leading()}
			{#if favicon}
				<img
					alt=""
					class="{ICON_CLASS_DEFAULT} shrink-0 rounded-sm"
					onerror={(e) => {
						(e.currentTarget as HTMLImageElement).style.display = 'none';
					}}
					src={favicon}
				/>
			{:else}
				{@const SourceIcon = TOOL_SOURCE_ICONS[group.source] ?? Server}
				<SourceIcon class="{ICON_CLASS_DEFAULT} shrink-0" />
			{/if}
		{/snippet}

		<span class="min-w-0 flex-1 truncate">{group.label}</span>

		<span class="text-body-sm text-muted-foreground tabular-nums">
			{toolsPanel.getEnabledToolCount(group)}/{group.tools.length}
		</span>

		{#snippet trailing()}
			{@render groupSwitch(
				group,
				isChecked
					? ATTACHMENT_MENU_TEXT.DISABLE_GROUP(group.label)
					: ATTACHMENT_MENU_TEXT.ENABLE_GROUP(group.label)
			)}
		{/snippet}
	</InsetListSplitRow>
{/snippet}

{#snippet groupPage(group: ToolGroup)}
	<InsetList class={toolsPanel.isGroupDisabled(group) ? 'opacity-50' : undefined}>
		<InsetListRow as="label" icon={PencilRuler}>
			<span class="flex-1">{ATTACHMENT_MENU_TEXT.ALL_TOOLS}</span>

			{@render groupSwitch(group)}
		</InsetListRow>
	</InsetList>

	<InsetList>
		{#each group.tools as entry (entry.key)}
			<InsetListRow
				as="label"
				class={toolsPanel.isToolParentDisabled(entry) ? 'opacity-50' : undefined}
			>
				<span class="min-w-0 flex-1 truncate font-mono text-body-sm">
					{entry.definition.function.name}
				</span>

				<Switch
					checked={toolsPanel.isToolEnabled(entry)}
					onCheckedChange={() => toolsPanel.toggleTool(entry)}
					size="lg"
				/>
			</InsetListRow>
		{/each}
	</InsetList>
{/snippet}
