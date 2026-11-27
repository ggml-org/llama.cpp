<script lang="ts">
	import SettingsChatImportDropZone from './SettingsChatImportDropZone.svelte';
	import SettingsChatImportExportSection from './SettingsChatImportExportSection.svelte';
	import { InsetList, InsetListRow } from '@ggml-org/llama-ui/app/sheets';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { Checkbox } from '@ggml-org/llama-ui/ui/checkbox';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import * as RadioGroup from '@ggml-org/llama-ui/ui/radio-group';
	import { Switch } from '@ggml-org/llama-ui/ui/switch';
	import { Check, Download, Trash2 } from '@lucide/svelte';
	import { DialogConfirmation, DialogConversationSelection } from '#lib/components/app/index.ts';
	import SettingsGroup from '#lib/components/app/settings/SettingsGroup.svelte';
	import { ICON_CLASS_DEFAULT } from '#lib/constants/index.ts';
	import { ConversationSelectionMode, DataExportScope } from '#lib/enums/index.ts';
	import { ConversationTransferService } from '#lib/services/index.ts';
	import { conversationsStore, settingsStore } from '#lib/stores/index.ts';
	import { createMessageCountMap } from '#lib/utils/index.ts';
	import { fade } from 'svelte/transition';
	import { toast } from 'svelte-sonner';

	interface Props {
		/** Draw the actions as a phone's inset list, for the settings sheet. */
		inset?: boolean;
	}

	let { inset = false }: Props = $props();

	const FADE_MS = 150;

	const TEXT = {
		DELETE: 'Delete',
		DELETE_ALL: 'Delete all conversations',
		DELETE_ALL_NOTE:
			'Permanently delete all conversations and their messages. This action cannot be undone. Consider exporting your conversations first if you want to keep a backup.',
		DELETE_ALL_TITLE: 'Delete All',
		EXPORT: 'Export',
		IMPORT: 'Import',
		IMPORT_NOTE:
			'Bring in one or more conversations, your settings, or both at once. You see what each file holds before anything changes; conversations already in your library are left as they are.',
		IMPORT_NOTE_SHORT:
			'Conversations, your settings or both, from a .jsonl, a .zip or a .json. You see what each file holds before anything changes.',
		INCLUDE_SENSITIVE: 'Include API keys and MCP server headers',
		SENSITIVE_INCLUDED:
			'The file will hold your credentials: share it with no one you do not trust.',
		SENSITIVE_LEFT_OUT: 'Left out by default to keep your credentials safe.'
	} as const;

	const EXPORT_OPTIONS = [
		{
			description: 'Pick which ones: one is saved as a JSONL file, several as a ZIP.',
			label: 'Conversations',
			value: DataExportScope.CONVERSATIONS
		},
		{
			description: 'Your chat settings and preferences as a JSON file.',
			label: 'Settings',
			value: DataExportScope.SETTINGS
		},
		{
			description: 'All conversations and settings in one ZIP, to restore in one drop.',
			label: 'Everything',
			value: DataExportScope.ALL
		}
	];

	let exportScope = $state(DataExportScope.CONVERSATIONS);
	let includeSensitiveData = $state(false);
	let exportsSettings = $derived(exportScope !== DataExportScope.CONVERSATIONS);

	let showExportDialog = $state(false);
	let availableConversations = $state<DatabaseConversation[]>([]);
	let messageCountMap = $state<Map<string, number>>(new Map());

	let showDeleteDialog = $state(false);

	async function handleExport() {
		try {
			if (exportScope === DataExportScope.SETTINGS) {
				ConversationTransferService.downloadSettingsFile(
					settingsStore.exportSettings(includeSensitiveData)
				);
				toast.success('Settings exported');
			} else if (exportScope === DataExportScope.ALL) {
				const data = await conversationsStore.getConversationsForExport(
					conversationsStore.conversations.map((conv) => conv.id)
				);

				ConversationTransferService.downloadConversationsArchive(
					data,
					settingsStore.exportSettings(includeSensitiveData)
				);
				toast.success(
					`Exported ${data.length} conversation${data.length === 1 ? '' : 's'} and settings`
				);
			} else {
				await openConversationPicker();
			}
		} catch (err) {
			console.error('Export failed:', err);
			toast.error('Export failed');
		}
	}

	async function openConversationPicker() {
		const allConversations = conversationsStore.conversations;

		if (allConversations.length === 0) {
			toast.info('No conversations to export');

			return;
		}

		const conversationsWithMessages = await Promise.all(
			allConversations.map(async (conv: DatabaseConversation) => ({
				conv,
				messages: await conversationsStore.getConversationMessages(conv.id)
			}))
		);

		messageCountMap = createMessageCountMap(conversationsWithMessages);
		availableConversations = allConversations;
		showExportDialog = true;
	}

	async function handleExportConfirm(selectedConversations: DatabaseConversation[]) {
		try {
			const allData = await conversationsStore.getConversationsForExport(
				selectedConversations.map((conv) => conv.id)
			);

			if (allData.length === 1) {
				ConversationTransferService.downloadConversationFile(allData[0]);
			} else {
				ConversationTransferService.downloadConversationsArchive(allData);
			}

			showExportDialog = false;
			toast.success(`Exported ${allData.length} conversation${allData.length === 1 ? '' : 's'}`);
		} catch (err) {
			console.error('Export failed:', err);
			toast.error('Failed to export conversations');
		}
	}

	function handleDeleteAllClick() {
		if (conversationsStore.conversations.length === 0) {
			toast.info('No conversations to delete');

			return;
		}

		showDeleteDialog = true;
	}

	async function handleDeleteAllConfirm() {
		try {
			await conversationsStore.deleteAll();

			showDeleteDialog = false;
		} catch (err) {
			console.error('Failed to delete conversations:', err);
		}
	}

	function handleDeleteAllCancel() {
		showDeleteDialog = false;
	}
</script>

{#snippet sensitiveNote()}
	{includeSensitiveData ? TEXT.SENSITIVE_INCLUDED : TEXT.SENSITIVE_LEFT_OUT}
{/snippet}

{#if inset}
	<div in:fade={{ duration: FADE_MS }} class="flex flex-col gap-8">
		<!-- the drop zone is its own surface, so it takes a group's title and caption, not its box -->
		<div class="flex flex-col gap-1.5">
			<h3 class="inset-list-title">{TEXT.IMPORT}</h3>

			<SettingsChatImportDropZone />

			<p class="inset-list-caption">{TEXT.IMPORT_NOTE_SHORT}</p>
		</div>

		<div class="flex flex-col gap-6">
			<InsetList title={TEXT.EXPORT}>
				{#each EXPORT_OPTIONS as option (option.value)}
					{@const selected = exportScope === option.value}

					<InsetListRow
						aria-pressed={selected}
						bodyClass="gap-3"
						onclick={() => (exportScope = option.value)}
					>
						<span class="flex flex-1 flex-col gap-0.5 py-3">
							<span>{option.label}</span>

							<span class="text-caption text-muted-foreground">{option.description}</span>
						</span>

						{#if selected}
							<Check class="size-4 shrink-0 text-(--action-primary)" />
						{/if}
					</InsetListRow>
				{/each}

				{#if exportsSettings}
					<InsetListRow as="label" bodyClass="gap-3">
						<span class="flex flex-1 flex-col gap-0.5 py-3">
							<span>{TEXT.INCLUDE_SENSITIVE}</span>

							<span
								class={[
									'text-caption',
									includeSensitiveData ? 'text-destructive' : 'text-muted-foreground'
								]}
							>
								{@render sensitiveNote()}
							</span>
						</span>

						<Switch bind:checked={includeSensitiveData} size="lg" />
					</InsetListRow>
				{/if}
			</InsetList>

			<InsetList>
				<InsetListRow icon={Download} onclick={handleExport}>{TEXT.EXPORT}</InsetListRow>
			</InsetList>
		</div>

		<InsetList title={TEXT.DELETE}>
			<SettingsChatImportExportSection
				IconComponent={Trash2}
				buttonText={TEXT.DELETE_ALL}
				buttonVariant="destructive"
				description={TEXT.DELETE_ALL_NOTE}
				inset
				onclick={handleDeleteAllClick}
				title={TEXT.DELETE_ALL_TITLE}
			/>
		</InsetList>
	</div>
{:else}
	<div in:fade={{ duration: FADE_MS }} class="space-y-12">
		<SettingsGroup title={TEXT.IMPORT}>
			<p class="text-body-sm text-muted-foreground">
				{TEXT.IMPORT_NOTE}
			</p>

			<SettingsChatImportDropZone />
		</SettingsGroup>

		<SettingsGroup title={TEXT.EXPORT}>
			<RadioGroup.Root
				class="gap-3"
				onValueChange={(value) => (exportScope = value as DataExportScope)}
				value={exportScope}
			>
				{#each EXPORT_OPTIONS as option (option.value)}
					{@const id = `export-${option.value}`}

					<div class="flex items-start gap-3">
						<RadioGroup.Item class="mt-0.5" {id} value={option.value} />

						<Label class="grid cursor-pointer gap-0.5" for={id}>
							<span class="text-label">{option.label}</span>

							<span class="text-caption font-normal text-muted-foreground"
								>{option.description}</span
							>
						</Label>
					</div>
				{/each}
			</RadioGroup.Root>

			{#if exportsSettings}
				<div transition:fade={{ duration: FADE_MS }} class="grid gap-1">
					<div class="flex items-center gap-2">
						<Checkbox bind:checked={includeSensitiveData} id="export-sensitive" />

						<Label class="text-body-sm font-normal" for="export-sensitive">
							{TEXT.INCLUDE_SENSITIVE}
						</Label>
					</div>

					<p
						class="text-caption {includeSensitiveData
							? 'text-destructive'
							: 'text-muted-foreground'}"
					>
						{@render sensitiveNote()}
					</p>
				</div>
			{/if}

			<Button class="justify-self-start" onclick={handleExport} variant="outline">
				<Download class="mr-2 {ICON_CLASS_DEFAULT}" />

				{TEXT.EXPORT}
			</Button>
		</SettingsGroup>

		<SettingsGroup title={TEXT.DELETE}>
			<SettingsChatImportExportSection
				IconComponent={Trash2}
				buttonClass="justify-start justify-self-start md:w-auto"
				buttonText={TEXT.DELETE_ALL}
				buttonVariant="destructive"
				description={TEXT.DELETE_ALL_NOTE}
				onclick={handleDeleteAllClick}
				title={TEXT.DELETE_ALL_TITLE}
				titleClass="text-destructive"
			/>
		</SettingsGroup>
	</div>
{/if}

<DialogConversationSelection
	bind:open={showExportDialog}
	conversations={availableConversations}
	{messageCountMap}
	mode={ConversationSelectionMode.EXPORT}
	onCancel={() => (showExportDialog = false)}
	onConfirm={handleExportConfirm}
/>

<DialogConfirmation
	bind:open={showDeleteDialog}
	cancelText="Cancel"
	confirmText="Delete All"
	description="Are you sure you want to delete all conversations? This action cannot be undone and will permanently remove all your conversations and messages."
	icon={Trash2}
	onCancel={handleDeleteAllCancel}
	onConfirm={handleDeleteAllConfirm}
	title={TEXT.DELETE_ALL}
	variant="destructive"
/>
