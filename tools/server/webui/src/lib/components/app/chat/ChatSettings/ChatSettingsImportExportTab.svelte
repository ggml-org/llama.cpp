<script lang="ts">
	import { Download, Upload, Trash2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { DialogConversationSelection } from '$lib/components/app';
	import { createMessageCountMap } from '$lib/utils';
	import { conversationsStore, conversations } from '$lib/stores/conversations.svelte';
	import { toast } from 'svelte-sonner';
	import DialogConfirmation from '$lib/components/app/dialogs/DialogConfirmation.svelte';
	import { t } from '$lib/i18n';

	let exportedConversations = $state<DatabaseConversation[]>([]);
	let importedConversations = $state<DatabaseConversation[]>([]);
	let showExportSummary = $state(false);
	let showImportSummary = $state(false);

	let showExportDialog = $state(false);
	let showImportDialog = $state(false);
	let availableConversations = $state<DatabaseConversation[]>([]);
	let messageCountMap = $state<Map<string, number>>(new Map());
	let fullImportData = $state<Array<{ conv: DatabaseConversation; messages: DatabaseMessage[] }>>(
		[]
	);

	// Delete functionality state
	let showDeleteDialog = $state(false);

	async function handleExportClick() {
		try {
			const allConversations = conversations();
			if (allConversations.length === 0) {
				toast.info(t('chat.settings.import_export.export.none'));
				return;
			}

			const conversationsWithMessages = await Promise.all(
				allConversations.map(async (conv: DatabaseConversation) => {
					const messages = await conversationsStore.getConversationMessages(conv.id);
					return { conv, messages };
				})
			);

			messageCountMap = createMessageCountMap(conversationsWithMessages);
			availableConversations = allConversations;
			showExportDialog = true;
		} catch (err) {
			console.error('Failed to load conversations:', err);
			alert(t('chat.settings.import_export.error.load_conversations'));
		}
	}

	async function handleExportConfirm(selectedConversations: DatabaseConversation[]) {
		try {
			const allData: ExportedConversations = await Promise.all(
				selectedConversations.map(async (conv) => {
					const messages = await conversationsStore.getConversationMessages(conv.id);
					return { conv: $state.snapshot(conv), messages: $state.snapshot(messages) };
				})
			);

			const blob = new Blob([JSON.stringify(allData, null, 2)], {
				type: 'application/json'
			});
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');

			a.href = url;
			a.download = `conversations_${new Date().toISOString().split('T')[0]}.json`;
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);

			exportedConversations = selectedConversations;
			showExportSummary = true;
			showImportSummary = false;
			showExportDialog = false;
		} catch (err) {
			console.error('Export failed:', err);
			alert(t('chat.settings.import_export.export.failed'));
		}
	}

	async function handleImportClick() {
		try {
			const input = document.createElement('input');

			input.type = 'file';
			input.accept = '.json';

			input.onchange = async (e) => {
				const file = (e.target as HTMLInputElement)?.files?.[0];
				if (!file) return;

				try {
					const text = await file.text();
					const parsedData = JSON.parse(text);
					let importedData: ExportedConversations;

					if (Array.isArray(parsedData)) {
						importedData = parsedData;
					} else if (
						parsedData &&
						typeof parsedData === 'object' &&
						'conv' in parsedData &&
						'messages' in parsedData
					) {
						// Single conversation object
						importedData = [parsedData];
					} else {
						throw new Error(t('chat.settings.import_export.import.invalid_format'));
					}

					fullImportData = importedData;
					availableConversations = importedData.map(
						(item: { conv: DatabaseConversation; messages: DatabaseMessage[] }) => item.conv
					);
					messageCountMap = createMessageCountMap(importedData);
					showImportDialog = true;
				} catch (err: unknown) {
					const message =
						err instanceof Error
							? err.message
							: t('chat.settings.import_export.error.unknown');

					console.error('Failed to parse file:', err);
					alert(t('chat.settings.import_export.import.parse_failed', { message }));
				}
			};

			input.click();
		} catch (err) {
			console.error('Import failed:', err);
			alert(t('chat.settings.import_export.import.failed'));
		}
	}

	async function handleImportConfirm(selectedConversations: DatabaseConversation[]) {
		try {
			const selectedIds = new Set(selectedConversations.map((c) => c.id));
			const selectedData = $state
				.snapshot(fullImportData)
				.filter((item) => selectedIds.has(item.conv.id));

			await conversationsStore.importConversationsData(selectedData);

			importedConversations = selectedConversations;
			showImportSummary = true;
			showExportSummary = false;
			showImportDialog = false;
		} catch (err) {
			console.error('Import failed:', err);
			alert(t('chat.settings.import_export.import.failed_format'));
		}
	}

	async function handleDeleteAllClick() {
		try {
			const allConversations = conversations();

			if (allConversations.length === 0) {
				toast.info(t('chat.settings.import_export.delete.none'));
				return;
			}

			showDeleteDialog = true;
		} catch (err) {
			console.error('Failed to load conversations for deletion:', err);
			toast.error(t('chat.settings.import_export.error.load_conversations'));
		}
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

<div class="space-y-6">
	<div class="space-y-4">
		<div class="grid">
			<h4 class="mb-2 text-sm font-medium">{t('chat.settings.import_export.export.title')}</h4>

			<p class="mb-4 text-sm text-muted-foreground">
				{t('chat.settings.import_export.export.description')}
			</p>

			<Button
				class="w-full justify-start justify-self-start md:w-auto"
				onclick={handleExportClick}
				variant="outline"
			>
				<Download class="mr-2 h-4 w-4" />

				{t('chat.settings.import_export.export.button')}
			</Button>

			{#if showExportSummary && exportedConversations.length > 0}
				<div class="mt-4 grid overflow-x-auto rounded-lg border border-border/50 bg-muted/30 p-4">
					<h5 class="mb-2 text-sm font-medium">
						{exportedConversations.length === 1
							? t('chat.settings.import_export.export.summary_one', {
									count: exportedConversations.length
								})
							: t('chat.settings.import_export.export.summary_many', {
									count: exportedConversations.length
								})}
					</h5>

					<ul class="space-y-1 text-sm text-muted-foreground">
						{#each exportedConversations.slice(0, 10) as conv (conv.id)}
							<li class="truncate">
								• {conv.name || t('dialog.conversation_selection.untitled')}
							</li>
						{/each}

						{#if exportedConversations.length > 10}
							<li class="italic">
								{t('chat.settings.import_export.summary.more', {
									count: exportedConversations.length - 10
								})}
							</li>
						{/if}
					</ul>
				</div>
			{/if}
		</div>

		<div class="grid border-t border-border/30 pt-4">
			<h4 class="mb-2 text-sm font-medium">{t('chat.settings.import_export.import.title')}</h4>

			<p class="mb-4 text-sm text-muted-foreground">
				{t('chat.settings.import_export.import.description')}
			</p>

			<Button
				class="w-full justify-start justify-self-start md:w-auto"
				onclick={handleImportClick}
				variant="outline"
			>
				<Upload class="mr-2 h-4 w-4" />
				{t('chat.settings.import_export.import.button')}
			</Button>

			{#if showImportSummary && importedConversations.length > 0}
				<div class="mt-4 grid overflow-x-auto rounded-lg border border-border/50 bg-muted/30 p-4">
					<h5 class="mb-2 text-sm font-medium">
						{importedConversations.length === 1
							? t('chat.settings.import_export.import.summary_one', {
									count: importedConversations.length
								})
							: t('chat.settings.import_export.import.summary_many', {
									count: importedConversations.length
								})}
					</h5>

					<ul class="space-y-1 text-sm text-muted-foreground">
						{#each importedConversations.slice(0, 10) as conv (conv.id)}
							<li class="truncate">
								• {conv.name || t('dialog.conversation_selection.untitled')}
							</li>
						{/each}

						{#if importedConversations.length > 10}
							<li class="italic">
								{t('chat.settings.import_export.summary.more', {
									count: importedConversations.length - 10
								})}
							</li>
						{/if}
					</ul>
				</div>
			{/if}
		</div>

		<div class="grid border-t border-border/30 pt-4">
			<h4 class="mb-2 text-sm font-medium text-destructive">
				{t('chat.settings.import_export.delete.title')}
			</h4>

			<p class="mb-4 text-sm text-muted-foreground">
				{t('chat.settings.import_export.delete.description')}
			</p>

			<Button
				class="text-destructive-foreground w-full justify-start justify-self-start bg-destructive hover:bg-destructive/80 md:w-auto"
				onclick={handleDeleteAllClick}
				variant="destructive"
			>
				<Trash2 class="mr-2 h-4 w-4" />

				{t('chat.settings.import_export.delete.button')}
			</Button>
		</div>
	</div>
</div>

<DialogConversationSelection
	conversations={availableConversations}
	{messageCountMap}
	mode="export"
	bind:open={showExportDialog}
	onCancel={() => (showExportDialog = false)}
	onConfirm={handleExportConfirm}
/>

<DialogConversationSelection
	conversations={availableConversations}
	{messageCountMap}
	mode="import"
	bind:open={showImportDialog}
	onCancel={() => (showImportDialog = false)}
	onConfirm={handleImportConfirm}
/>

<DialogConfirmation
	bind:open={showDeleteDialog}
	title={t('chat.settings.import_export.delete.dialog.title')}
	description={t('chat.settings.import_export.delete.dialog.description')}
	confirmText={t('chat.settings.import_export.delete.dialog.confirm')}
	cancelText={t('chat.settings.import_export.delete.dialog.cancel')}
	variant="destructive"
	icon={Trash2}
	onConfirm={handleDeleteAllConfirm}
	onCancel={handleDeleteAllCancel}
/>
