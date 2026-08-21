/**
 * Helpers for deciding whether a dropped file is an import file
 * (conversation/settings export) rather than a plain attachment.
 *
 * The chat screen attaches arbitrary files to messages, while the global
 * drag-and-drop handler imports conversations and settings. Exported
 * conversations are `.jsonl` (single) or `.zip` (archive), and settings are
 * `.json`, so the extension is a reliable way to route a drop to the importer.
 */

const IMPORT_FILE_EXTENSIONS = ['.zip', '.jsonl', '.json'];

export function isImportFileByExtension(file: File): boolean {
	const name = file.name.toLowerCase();

	return IMPORT_FILE_EXTENSIONS.some((ext) => name.endsWith(ext));
}
