<script lang="ts">
	import { File as FileIcon, Paperclip, Plus, Server, X } from '@lucide/svelte';
	import { ActionIcon, ChatFormPickerMention, DropdownMenuActions } from '$lib/components/app';
	import { FileMentionEntryType, MentionPickerSource, ProjectContextKind } from '$lib/enums';
	import { useChatScreenActiveModel } from '$lib/hooks/use-chat-screen-active-model.svelte';
	import type { FileMentionEntry, ProjectContextItem } from '$lib/types';
	import { formatFileSize, uuid } from '$lib/utils';
	import { parseFilesToMessageExtras, processFilesToChatUploaded } from '$lib/utils/browser-only';

	interface Props {
		items: ProjectContextItem[];
	}

	let { items = $bindable() }: Props = $props();

	const activeModel = useChatScreenActiveModel();

	let fileInput = $state<HTMLInputElement | null>(null);
	let pickerAnchor = $state<HTMLDivElement | null>(null);
	let pickerSource = $state<MentionPickerSource>(MentionPickerSource.FILES);
	let isPickerOpen = $state(false);
	let pickerQuery = $state('');

	const actions = [
		{ icon: Paperclip, label: 'Upload files', onclick: () => fileInput?.click() },
		{
			icon: FileIcon,
			label: 'Reference files',
			onclick: () => openPicker(MentionPickerSource.FILES),
			separator: true
		},
		{
			icon: Server,
			label: 'MCP resources',
			onclick: () => openPicker(MentionPickerSource.MCP_RESOURCES)
		}
	];

	function openPicker(source: MentionPickerSource) {
		pickerSource = source;
		pickerQuery = '';
		isPickerOpen = true;
	}

	/**
	 * Uploaded files are parsed the same way chat attachments are, so a project
	 * chat can put their content in front of the model.
	 */
	async function handleFileInput(event: Event & { currentTarget: HTMLInputElement }) {
		const files = Array.from(event.currentTarget.files ?? []);

		event.currentTarget.value = '';

		if (files.length === 0) return;

		const modelId = activeModel.activeModelId ?? undefined;
		const uploaded = await processFilesToChatUploaded(files, modelId);
		const { extras } = await parseFilesToMessageExtras($state.snapshot(uploaded), modelId);

		items = [
			...items,
			...extras.map((extra) => ({
				extra,
				id: uuid(),
				kind: ProjectContextKind.ATTACHMENT,
				name: 'name' in extra ? String(extra.name) : 'attachment',
				size: 'size' in extra ? extra.size : undefined
			}))
		];
	}

	/** A file or MCP resource picked from the server is kept as a reference. */
	function handleReferenceSelect(entry: FileMentionEntry) {
		isPickerOpen = false;

		items = [
			...items,
			{
				id: uuid(),
				kind:
					entry.type === FileMentionEntryType.MCP_RESOURCE
						? ProjectContextKind.MCP_RESOURCE
						: ProjectContextKind.FILE,
				name: entry.name,
				path: entry.path,
				serverName: entry.serverName
			}
		];
	}

	function itemIcon(kind: ProjectContextKind) {
		if (kind === ProjectContextKind.ATTACHMENT) return Paperclip;

		if (kind === ProjectContextKind.MCP_RESOURCE) return Server;

		return FileIcon;
	}

	function itemDetail(item: ProjectContextItem): string {
		if (item.kind === ProjectContextKind.ATTACHMENT) return formatFileSize(item.size ?? 0);

		if (item.kind === ProjectContextKind.MCP_RESOURCE) return item.serverName ?? '';

		return item.path ?? '';
	}
</script>

<div bind:this={pickerAnchor} class="flex flex-col gap-1">
	<div class="flex items-center justify-between gap-2">
		<span class="text-sm font-medium">Context</span>

		<DropdownMenuActions
			{actions}
			triggerClass="h-6 w-6"
			triggerIcon={Plus}
			triggerTooltip="Add context"
		/>
	</div>

	<input bind:this={fileInput} class="hidden" multiple onchange={handleFileInput} type="file" />

	{#if items.length === 0}
		<p class="text-sm text-muted-foreground">
			Upload files, reference files on the server, or attach MCP resources
		</p>
	{:else}
		{#each items as item (item.id)}
			{@const Icon = itemIcon(item.kind)}

			<div class="group flex items-center gap-2 rounded-md px-1 py-1 hover:bg-muted/40">
				<Icon class="h-4 w-4 shrink-0 text-muted-foreground" />

				<span class="min-w-0 flex-1 truncate text-sm" title={item.path ?? item.name}>
					{item.name}
				</span>

				<span class="max-w-32 shrink-0 truncate text-xs text-muted-foreground tabular-nums">
					{itemDetail(item)}
				</span>

				<ActionIcon
					ariaLabel="Remove {item.name}"
					class="h-5 w-5 shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
					icon={X}
					iconSize="h-3.5 w-3.5"
					onclick={() => (items = items.filter((other) => other.id !== item.id))}
					tooltip="Remove from context"
					tooltipAsTitle
				/>
			</div>
		{/each}
	{/if}
</div>

<ChatFormPickerMention
	bind:query={pickerQuery}
	customAnchor={pickerAnchor}
	isOpen={isPickerOpen}
	onClose={() => (isPickerOpen = false)}
	onSelect={handleReferenceSelect}
	showSearchInput
	source={pickerSource}
/>
