/**
 * Global drag-and-drop import state machine.
 *
 * Tracks pointer enter/leave nesting so the overlay stays visible while the
 * cursor traverses child elements, then routes dropped files to the right
 * importer: settings files (JSON with a `config` key) are restored directly,
 * while conversation files (JSONL/ZIP/JSON) go through the existing
 * selection dialog before asking whether to open the result.
 */

import type { SettingsDiffEntry } from '#lib/components/app/dialogs/DialogSettingsImportPreview.svelte';
import { SETTING_CONFIG_DEFAULT, SETTINGS_REGISTRY, ZIP_MAGIC } from '#lib/constants/index.ts';
import { ConversationTransferService } from '#lib/services/conversation-transfer.service.ts';
import { RouterService } from '#lib/services/router.service.ts';
import { conversationsStore } from '#lib/stores/conversations/index.svelte.ts';
import { settingsStore } from '#lib/stores/settings/index.svelte.ts';
import type { SettingsConfigType, SettingsExportType } from '#lib/types/index.ts';
import { createMessageCountMap } from '#lib/utils/index.ts';
import { goto } from '$app/navigation';
import { strFromU8 } from 'fflate';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';

type FileKind = 'settings' | 'conversations';

/**
 * Detects whether a dropped file holds settings or conversations.
 * Settings files are JSON objects carrying a `config` key; everything else
 * (ZIP archives, JSONL sessions, legacy JSON) is treated as conversations.
 */
async function classifyFile(file: File): Promise<FileKind> {
	const bytes = new Uint8Array(await file.arrayBuffer());

	if (ZIP_MAGIC.every((byte, index) => bytes[index] === byte)) {
		return 'conversations';
	}

	const text = strFromU8(bytes);

	try {
		const parsed = JSON.parse(text);

		if (parsed && typeof parsed === 'object' && 'config' in parsed) {
			return 'settings';
		}
	} catch {
		// Not a JSON object, so not a settings file.
	}

	return 'conversations';
}

/**
 * Collapses several exports of the same conversation (e.g. the same chat
 * exported twice) to the most recently modified one, keeping the drop order.
 */
function keepNewestPerConversation(items: ExportedConversation[]): ExportedConversation[] {
	const newest: Record<string, ExportedConversation> = {};

	for (const item of items) {
		const current = newest[item.conv.id];

		if (!current || item.conv.lastModified > current.conv.lastModified) {
			newest[item.conv.id] = item;
		}
	}

	return items.filter((item) => newest[item.conv.id] === item);
}

export function computeSettingsDiff(
	current: SettingsConfigType,
	imported: SettingsConfigType
): SettingsDiffEntry[] {
	const labels = new SvelteMap<string, string>();

	for (const section of SETTINGS_REGISTRY) {
		for (const setting of section.settings) {
			labels.set(setting.key, setting.label);
		}
	}

	// importing applies the file over the defaults, so a key the file leaves out goes back to its default
	const next: SettingsConfigType = { ...SETTING_CONFIG_DEFAULT, ...imported };
	const keys = new SvelteSet([...Object.keys(current), ...Object.keys(next)]);
	const diff: SettingsDiffEntry[] = [];

	for (const key of keys) {
		const from = current[key];
		const to = next[key];

		if (from !== to) {
			diff.push({ from, key, label: labels.get(key) ?? key, to });
		}
	}

	return diff;
}

export function useDropImport() {
	let dragCounter = $state(0);
	let isDragOver = $state(false);

	// All dialog state lives in one reactive object so it stays reactive when
	// exposed through the hook and bound from the layout.
	const ui = $state({
		availableConversations: [] as DatabaseConversation[],
		bulkMessageCountMap: new SvelteMap() as SvelteMap<string, number>,
		fullImportData: [] as ExportedConversation[],
		importedConversations: [] as DatabaseConversation[],
		// The single conversation just imported, offered to open right away.
		openImported: null as { alreadyInLibrary: boolean; conv: DatabaseConversation } | null,
		previewData: null as ExportedConversation | null,
		selectionMessageCountMap: new SvelteMap() as SvelteMap<string, number>,
		settingsData: null as SettingsExportType | null,
		settingsDiff: [] as SettingsDiffEntry[],
		// Bulk result dialog: pick one of the imported conversations to open.
		showOpenBulk: false,
		// Asks whether to open the single conversation that was just imported.
		showOpenImported: false,
		// Single conversation preview dialog (confirm before importing).
		showPreview: false,
		// Selection dialog (the existing import flow) for multiple conversations.
		showSelection: false,
		// Settings import preview dialog (review diff before applying).
		showSettingsPreview: false
	});

	function handleDragEnter(event: DragEvent) {
		event.preventDefault();
		dragCounter++;

		if (event.dataTransfer?.types.includes('Files')) {
			isDragOver = true;
		}
	}

	function handleDragLeave(event: DragEvent) {
		event.preventDefault();
		dragCounter--;

		if (dragCounter === 0) {
			isDragOver = false;
		}
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
	}

	async function handleDrop(event: DragEvent) {
		event.preventDefault();
		isDragOver = false;
		dragCounter = 0;

		if (!event.dataTransfer?.files) return;

		const files = Array.from(event.dataTransfer.files);

		await processFiles(files);
	}

	async function processFiles(files: File[]) {
		const allConversations: ExportedConversation[] = [];

		for (const file of files) {
			const kind = await classifyFile(file);

			if (kind === 'settings') {
				try {
					const data = JSON.parse(await file.text());

					if (data?.config) {
						ui.settingsData = data as SettingsExportType;
						ui.settingsDiff = computeSettingsDiff(
							$state.snapshot(settingsStore.config) as SettingsConfigType,
							data.config
						);
						ui.showSettingsPreview = true;
					} else {
						toast.error(`Invalid settings file: ${file.name}`);
					}
				} catch (err) {
					console.error('Failed to import settings:', err);
					toast.error(`Failed to import settings from ${file.name}`);
				}
			} else {
				try {
					const parsed = await ConversationTransferService.parseImportFile(file);

					allConversations.push(...parsed);
				} catch (err) {
					console.error('Failed to parse file:', err);
					toast.error(`Failed to parse ${file.name}`);
				}
			}
		}

		if (allConversations.length === 0) {
			return;
		}

		const conversations = keepNewestPerConversation(allConversations);
		const duplicates = allConversations.length - conversations.length;

		if (duplicates > 0) {
			toast.info(
				duplicates === 1
					? 'The same conversation was dropped twice, keeping the newest copy'
					: `${duplicates} conversations were dropped more than once, keeping the newest copies`
			);
		}

		if (conversations.length === 1) {
			ui.previewData = conversations[0];
			ui.showPreview = true;
		} else {
			ui.fullImportData = conversations;
			ui.availableConversations = conversations.map((item) => item.conv);
			ui.selectionMessageCountMap = new SvelteMap(createMessageCountMap(conversations));
			ui.showSelection = true;
		}
	}

	async function confirmSettingsImport() {
		if (!ui.settingsData) return;

		const data = $state.snapshot(ui.settingsData) as SettingsExportType;

		try {
			settingsStore.importSettings(data);
			ui.showSettingsPreview = false;
			toast.success('Settings imported successfully');
		} catch (err) {
			console.error('Failed to import settings:', err);
			toast.error('Failed to import settings');
		}
	}

	function cancelSettingsImport() {
		ui.showSettingsPreview = false;
	}

	async function confirmImportSingle() {
		if (!ui.previewData) return;

		// IndexedDB cannot clone the reactive proxy, so store a plain copy.
		const data = $state.snapshot(ui.previewData) as ExportedConversation;

		try {
			const { skipped } = await conversationsStore.importConversationsData([data]);

			ui.showPreview = false;
			ui.openImported = { alreadyInLibrary: skipped.length > 0, conv: data.conv };
			ui.showOpenImported = true;
		} catch (err) {
			console.error('Failed to import conversation:', err);
			toast.error('Failed to import conversation');
		}
	}

	async function handleSelectionConfirm(selectedConversations: DatabaseConversation[]) {
		try {
			const selectedIds = new SvelteSet(selectedConversations.map((c) => c.id));
			const selectedData = ($state.snapshot(ui.fullImportData) as ExportedConversation[]).filter(
				(item) => selectedIds.has(item.conv.id)
			);

			await conversationsStore.importConversationsData(selectedData);

			ui.importedConversations = selectedConversations;
			ui.bulkMessageCountMap = new SvelteMap(createMessageCountMap(selectedData));
			ui.showSelection = false;
			ui.showOpenBulk = true;
		} catch (err) {
			console.error('Import failed:', err);
			toast.error('Failed to import conversations');
		}
	}

	function openConversation(conversation: DatabaseConversation) {
		ui.showOpenBulk = false;
		goto(RouterService.chat(conversation.id));
	}

	function confirmOpenImported() {
		const imported = ui.openImported;

		ui.showOpenImported = false;

		if (imported) goto(RouterService.chat(imported.conv.id));
	}

	function cancelOpenImported() {
		ui.showOpenImported = false;
	}

	function cancelPreview() {
		ui.showPreview = false;
	}

	function cancelBulk() {
		ui.showOpenBulk = false;
	}

	function cancelSelection() {
		ui.showSelection = false;
	}

	return {
		cancelBulk,
		cancelOpenImported,
		cancelPreview,
		cancelSelection,
		cancelSettingsImport,
		confirmImportSingle,
		confirmOpenImported,
		confirmSettingsImport,
		dragHandlers: {
			dragenter: handleDragEnter,
			dragleave: handleDragLeave,
			dragover: handleDragOver,
			drop: handleDrop
		},
		handleSelectionConfirm,
		get isDragOver() {
			return isDragOver;
		},
		openConversation,
		ui
	};
}
