<script lang="ts">
	import type { Component } from 'svelte';
	import { Download, Upload, Trash2, Database } from '@lucide/svelte';
	import { Button, type ButtonVariant } from '$lib/components/ui/button';
	import { DialogConversationSelection, DialogConfirmation } from '$lib/components/app';
	import { createMessageCountMap } from '$lib/utils';
	import { conversationsStore, conversations } from '$lib/stores/conversations.svelte';
	import { toast } from 'svelte-sonner';
	import { fade } from 'svelte/transition';
	import { ConversationSelectionMode, HtmlInputType, FileExtensionText } from '$lib/enums';

	interface SectionOpts {
		wrapperClass?: string;
		titleClass?: string;
		buttonVariant?: ButtonVariant;
		buttonClass?: string;
		summary?: { show: boolean; verb: string; items: DatabaseConversation[] };
	}

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
				toast.info('No conversations to export');
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
			alert('Failed to load conversations');
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

			conversationsStore.downloadConversationFile(allData);

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

			input.type = HtmlInputType.FILE;
			input.accept = FileExtensionText.JSON;

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

			await conversationsStore.importConversationsData(selectedData);

			importedConversations = selectedConversations;
			showImportSummary = true;
			showExportSummary = false;
			showImportDialog = false;
		} catch (err) {
			console.error('Import failed:', err);
			alert('Failed to import conversations. Please check the file format.');
		}
	}

	async function handleDeleteAllClick() {
		try {
			const allConversations = conversations();

			if (allConversations.length === 0) {
				toast.info('No conversations to delete');
				return;
			}

			showDeleteDialog = true;
		} catch (err) {
			console.error('Failed to load conversations for deletion:', err);
			toast.error('Failed to load conversations');
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

{#snippet summaryList(show: boolean, verb: string, items: DatabaseConversation[])}
	{#if show && items.length > 0}
		<div class="mt-4 grid overflow-x-auto rounded-lg border border-border/50 bg-muted/30 p-4">
			<h5 class="mb-2 text-sm font-medium">
				{verb}
				{items.length} conversation{items.length === 1 ? '' : 's'}
			</h5>

			<ul class="space-y-1 text-sm text-muted-foreground">
				{#each items.slice(0, 10) as conv (conv.id)}
					<li class="truncate">• {conv.name || 'Untitled conversation'}</li>
				{/each}

				{#if items.length > 10}
					<li class="italic">... and {items.length - 10} more</li>
				{/if}
			</ul>
		</div>
	{/if}
{/snippet}

{#snippet section(
	title: string,
	description: string,
	Icon: Component,
	buttonText: string,
	onclick: () => void,
	opts: SectionOpts
)}
	{@const buttonClass = opts?.buttonClass ?? 'justify-start justify-self-start md:w-auto'}
	{@const buttonVariant = opts?.buttonVariant ?? 'outline'}
	<div class="grid gap-1 {opts?.wrapperClass ?? ''}">
		<h4 class="mt-0 mb-2 text-sm font-medium {opts?.titleClass ?? ''}">{title}</h4>

		<p class="mb-4 text-sm text-muted-foreground">{description}</p>

		<Button class={buttonClass} {onclick} variant={buttonVariant}>
			<Icon class="mr-2 h-4 w-4" />

			{buttonText}
		</Button>

		{#if opts?.summary}
			{@render summaryList(opts.summary.show, opts.summary.verb, opts.summary.items)}
		{/if}
	</div>
{/snippet}

<div class="space-y-6" in:fade={{ duration: 150 }}>
	<div class="flex items-center gap-2 pb-4">
		<Database class="h-5 w-5 md:h-6 md:w-6" />

		<h1 class="text-xl font-semibold md:text-2xl">Import / Export</h1>
	</div>

	<div class="space-y-6">
		{@render section(
			'Export Conversations',
			'Download all your conversations as a JSON file. This includes all messages, attachments, and conversation history.',
			Download,
			'Export conversations',
			handleExportClick,
			{ summary: { show: showExportSummary, verb: 'Exported', items: exportedConversations } }
		)}

		{@render section(
			'Import Conversations',
			'Import one or more conversations from a previously exported JSON file. This will merge with your existing conversations.',
			Upload,
			'Import conversations',
			handleImportClick,
			{
				wrapperClass: 'border-t border-border/30 pt-6',
				summary: { show: showImportSummary, verb: 'Imported', items: importedConversations }
			}
		)}

		{@render section(
			'Delete All Conversations',
			'Permanently delete all conversations and their messages. This action cannot be undone. Consider exporting your conversations first if you want to keep a backup.',
			Trash2,
			'Delete all conversations',
			handleDeleteAllClick,
			{
				wrapperClass: 'border-t border-border/30 pt-4',
				titleClass: 'text-destructive',
				buttonVariant: 'destructive',
				buttonClass:
					'text-destructive-foreground justify-start justify-self-start bg-destructive hover:bg-destructive/80 md:w-auto'
			}
		)}
	</div>
</div>

<DialogConversationSelection
	conversations={availableConversations}
	{messageCountMap}
	mode={ConversationSelectionMode.EXPORT}
	bind:open={showExportDialog}
	onCancel={() => (showExportDialog = false)}
	onConfirm={handleExportConfirm}
/>

<DialogConversationSelection
	conversations={availableConversations}
	{messageCountMap}
	mode={ConversationSelectionMode.IMPORT}
	bind:open={showImportDialog}
	onCancel={() => (showImportDialog = false)}
	onConfirm={handleImportConfirm}
/>

<DialogConfirmation
	bind:open={showDeleteDialog}
	title="Delete all conversations"
	description="Are you sure you want to delete all conversations? This action cannot be undone and will permanently remove all your conversations and messages."
	confirmText="Delete All"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleDeleteAllConfirm}
	onCancel={handleDeleteAllCancel}
/>
