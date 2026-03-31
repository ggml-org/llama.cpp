import { getContext, setContext } from 'svelte';
import { CONTEXT_KEY_IMPORT_EXPORT_DIALOG } from '$lib/constants';

export interface ImportExportDialogContext {
	open: () => void;
}

const IMPORT_EXPORT_DIALOG_KEY = Symbol.for(CONTEXT_KEY_IMPORT_EXPORT_DIALOG);

export function setImportExportDialogContext(
	ctx: ImportExportDialogContext
): ImportExportDialogContext {
	return setContext(IMPORT_EXPORT_DIALOG_KEY, ctx);
}

export function getImportExportDialogContext(): ImportExportDialogContext {
	return getContext(IMPORT_EXPORT_DIALOG_KEY);
}
