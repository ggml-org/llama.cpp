<script lang="ts">
	import { Download, Upload } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import ConversationSelectionDialog from './ConversationSelectionDialog.svelte';
	import { DatabaseStore } from '$lib/stores/database';
	import type { ExportedConversations } from '$lib/types/database';
	import { createMessageCountMap } from '$lib/utils/conversation-utils';
	import { chatStore } from '$lib/stores/chat.svelte';

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

	async function handleExportClick() {
		try {
			const allConversations = await DatabaseStore.getAllConversations();
			if (allConversations.length === 0) {
				alert('No conversations to export');
				return;
			}

			const conversationsWithMessages = await Promise.all(
				allConversations.map(async (conv) => {
					const messages = await DatabaseStore.getConversationMessages(conv.id);
					return { conv, messages };
				})
			);

			messageCountMap = createMessageCountMap(conversationsWithMessages);
			availableConversations = allConversations;
			showExportDialog = true;
		} catch (err) {
			console.error('Failed to load conversations:', err);
			alert('Failed to load conversations');
		}
	}

	async function handleExportConfirm(selectedConversations: DatabaseConversation[]) {
		try {
			const allData: ExportedConversations = await Promise.all(
				selectedConversations.map(async (conv) => {
					const messages = await DatabaseStore.getConversationMessages(conv.id);
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
			alert('Failed to export conversations');
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
						throw new Error(
							'Invalid file format: expected array of conversations or single conversation object'
						);
					}

					fullImportData = importedData;
					availableConversations = importedData.map(
						(item: { conv: DatabaseConversation; messages: DatabaseMessage[] }) => item.conv
					);
					messageCountMap = createMessageCountMap(importedData);
					showImportDialog = true;
				} catch (err: unknown) {
					const message = err instanceof Error ? err.message : 'Unknown error';

					console.error('Failed to parse file:', err);
					alert(`Failed to parse file: ${message}`);
				}
			};

			input.click();
		} catch (err) {
			console.error('Import failed:', err);
			alert('Failed to import conversations');
		}
	}

	async function handleImportConfirm(selectedConversations: DatabaseConversation[]) {
		try {
			const selectedIds = new Set(selectedConversations.map((c) => c.id));
			const selectedData = $state
				.snapshot(fullImportData)
				.filter((item) => selectedIds.has(item.conv.id));

			await DatabaseStore.importConversations(selectedData);

			await chatStore.loadConversations();

			importedConversations = selectedConversations;
			showImportSummary = true;
			showExportSummary = false;
			showImportDialog = false;
		} catch (err) {
			console.error('Import failed:', err);
			alert('Failed to import conversations. Please check the file format.');
		}
	}
</script>

<div class="space-y-6">
	<div class="space-y-4">
		<div class="grid">
			<h4 class="mb-2 text-sm font-medium">Export Conversations</h4>

			<p class="mb-4 text-sm text-muted-foreground">
				Download all your conversations as a JSON file. This includes all messages, attachments, and
				conversation history.
			</p>

			<Button
				class="w-full justify-start justify-self-start md:w-auto"
				onclick={handleExportClick}
				variant="outline"
			>
				<Download class="mr-2 h-4 w-4" />

				Export conversations
			</Button>

			{#if showExportSummary && exportedConversations.length > 0}
				<div class="mt-4 grid overflow-x-auto rounded-lg border border-border/50 bg-muted/30 p-4">
					<h5 class="mb-2 text-sm font-medium">
						Exported {exportedConversations.length} conversation{exportedConversations.length === 1
							? ''
							: 's'}
					</h5>

					<ul class="space-y-1 text-sm text-muted-foreground">
						{#each exportedConversations.slice(0, 10) as conv (conv.id)}
							<li class="truncate">• {conv.name || 'Untitled conversation'}</li>
						{/each}

						{#if exportedConversations.length > 10}
							<li class="italic">
								... and {exportedConversations.length - 10} more
							</li>
						{/if}
					</ul>
				</div>
			{/if}
		</div>

		<div class="grid border-t border-border/30 pt-4">
			<h4 class="mb-2 text-sm font-medium">Import Conversations</h4>

			<p class="mb-4 text-sm text-muted-foreground">
				Import one or more conversations from a previously exported JSON file. This will merge with
				your existing conversations.
			</p>

			<Button
				class="w-full justify-start justify-self-start md:w-auto"
				onclick={handleImportClick}
				variant="outline"
			>
				<Upload class="mr-2 h-4 w-4" />
				Import conversations
			</Button>

			{#if showImportSummary && importedConversations.length > 0}
				<div class="mt-4 grid overflow-x-auto rounded-lg border border-border/50 bg-muted/30 p-4">
					<h5 class="mb-2 text-sm font-medium">
						Imported {importedConversations.length} conversation{importedConversations.length === 1
							? ''
							: 's'}
					</h5>

					<ul class="space-y-1 text-sm text-muted-foreground">
						{#each importedConversations.slice(0, 10) as conv (conv.id)}
							<li class="truncate">• {conv.name || 'Untitled conversation'}</li>
						{/each}

						{#if importedConversations.length > 10}
							<li class="italic">
								... and {importedConversations.length - 10} more
							</li>
						{/if}
					</ul>
				</div>
			{/if}
		</div>
	</div>
</div>

<ConversationSelectionDialog
	conversations={availableConversations}
	{messageCountMap}
	mode="export"
	bind:open={showExportDialog}
	onCancel={() => (showExportDialog = false)}
	onConfirm={handleExportConfirm}
/>

<ConversationSelectionDialog
	conversations={availableConversations}
	{messageCountMap}
	mode="import"
	bind:open={showImportDialog}
	onCancel={() => (showImportDialog = false)}
	onConfirm={handleImportConfirm}
/>
