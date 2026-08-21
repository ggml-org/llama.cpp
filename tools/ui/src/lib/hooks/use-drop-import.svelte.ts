/**
 * Global drag-and-drop import state machine.
 *
 * Tracks pointer enter/leave nesting so the overlay stays visible while the
 * cursor traverses child elements, then routes dropped files to the right
 * importer: settings files (JSON with a `config` key) are restored directly,
 * while conversation files (JSONL/ZIP/JSON) go through the existing
 * selection dialog before asking whether to open the result.
 */

import { goto } from '$app/navigation';
import type { SettingsDiffEntry } from '$lib/components/app/dialogs/DialogSettingsImportPreview.svelte';
import { ZIP_MAGIC } from '$lib/constants';
import { SETTINGS_REGISTRY } from '$lib/constants/settings.constants';
import { ConversationTransferService, RouterService } from '$lib/services';
import { conversationsStore, settingsStore } from '$lib/stores';
import type { SettingsConfigType, SettingsExportType } from '$lib/types';
import { createMessageCountMap } from '$lib/utils';
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

	const keys = new SvelteSet([...Object.keys(current), ...Object.keys(imported)]);
	const diff: SettingsDiffEntry[] = [];

	for (const key of keys) {
		const from = current[key];
		const to = imported[key];

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
		previewData: null as ExportedConversation | null,
		selectionMessageCountMap: new SvelteMap() as SvelteMap<string, number>,
		settingsData: null as SettingsExportType | null,
		settingsDiff: [] as SettingsDiffEntry[],
		// Bulk result dialog: pick one of the imported conversations to open.
		showOpenBulk: false,
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

		if (allConversations.length === 1) {
			ui.previewData = allConversations[0];
			ui.showPreview = true;
		} else {
			ui.fullImportData = allConversations;
			ui.availableConversations = allConversations.map((item) => item.conv);
			ui.selectionMessageCountMap = new SvelteMap(createMessageCountMap(allConversations));
			ui.showSelection = true;
		}
	}

	async function confirmSettingsImport() {
		const data = ui.settingsData;

		if (!data) return;

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
		const data = ui.previewData;

		if (!data) return;

		try {
			await conversationsStore.importConversationsData([data]);

			ui.showPreview = false;
			goto(RouterService.chat(data.conv.id));
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
		cancelPreview,
		cancelSelection,
		cancelSettingsImport,
		confirmImportSingle,
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
