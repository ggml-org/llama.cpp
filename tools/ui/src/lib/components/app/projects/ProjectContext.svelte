<script lang="ts">
	import ProjectSection from './ProjectSection.svelte';
	import { File as FileIcon, Paperclip, Plus, Server, X } from '@lucide/svelte';
	import { ActionIcon, ChatFormPickerMention, DropdownMenuActions } from '$lib/components/app';
	import * as Empty from '$lib/components/ui/empty';
	import { ICON_CLASS_DEFAULT, SECTION_ACTION_CLASS } from '$lib/constants';
	import { MentionPickerSource, ProjectContextKind } from '$lib/enums';
	import type { FileMentionEntry, ProjectContextItem } from '$lib/types';
	import { formatFileSize } from '$lib/utils';

	interface Props {
		items: ProjectContextItem[];
		onAddAttachments: (files: File[]) => void;
		onAddReference: (entry: FileMentionEntry) => void;
		onRemove: (id: string) => void;
	}

	let { items, onAddAttachments, onAddReference, onRemove }: Props = $props();

	let fileInput = $state<HTMLInputElement | null>(null);
	let pickerAnchor = $state<HTMLDivElement | null>(null);
	let pickerSource = $state<MentionPickerSource>(MentionPickerSource.FILES);
	let isPickerOpen = $state(false);
	let pickerQuery = $state('');

	const actions = $derived([
		{
			icon: Paperclip,
			label: 'Upload files',
			onclick: () => fileInput?.click()
		},
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
	]);

	function openPicker(source: MentionPickerSource) {
		pickerSource = source;
		pickerQuery = '';
		isPickerOpen = true;
	}

	function handleFileInput(event: Event & { currentTarget: HTMLInputElement }) {
		const files = Array.from(event.currentTarget.files ?? []);

		if (files.length > 0) onAddAttachments(files);

		event.currentTarget.value = '';
	}

	function handleReferenceSelect(entry: FileMentionEntry) {
		onAddReference(entry);
		isPickerOpen = false;
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

<div bind:this={pickerAnchor}>
	<ProjectSection title="Context">
		{#snippet action()}
			<DropdownMenuActions
				{actions}
				triggerClass={SECTION_ACTION_CLASS}
				triggerIcon={Plus}
				triggerIconClass={ICON_CLASS_DEFAULT}
				triggerTooltip="Add context to the project"
			/>
		{/snippet}

		<input bind:this={fileInput} class="hidden" multiple onchange={handleFileInput} type="file" />

		{#if items.length === 0}
			<Empty.Root class="gap-2 p-0">
				<Empty.Header class="gap-1">
					<Empty.Media variant="icon">
						<FileIcon />
					</Empty.Media>

					<Empty.Description class="text-sm text-balance">
						Upload files, reference files on the server, or attach MCP resources.
					</Empty.Description>
				</Empty.Header>
			</Empty.Root>
		{:else}
			<div class="flex flex-col gap-1">
				{#each items as item (item.id)}
					{@const Icon = itemIcon(item.kind)}

					<div class="group flex items-center gap-2 rounded-md px-1 py-1.5 hover:bg-muted/40">
						<Icon class="h-4 w-4 shrink-0 text-muted-foreground" />

						<span class="min-w-0 flex-1 truncate text-sm" title={item.path ?? item.name}>
							{item.name}
						</span>

						<span class="max-w-32 shrink-0 truncate text-xs text-muted-foreground tabular-nums">
							{itemDetail(item)}
						</span>

						<ActionIcon
							ariaLabel="Remove {item.name}"
							class="h-5 w-5 shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100"
							icon={X}
							iconSize="h-3.5 w-3.5"
							onclick={() => onRemove(item.id)}
							tooltip="Remove from context"
							tooltipAsTitle
						/>
					</div>
				{/each}
			</div>
		{/if}
	</ProjectSection>
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
