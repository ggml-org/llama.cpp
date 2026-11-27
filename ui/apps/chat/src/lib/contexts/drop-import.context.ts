import { getContext, setContext } from 'svelte';

export interface DropImportContext {
	/** Imports the dropped files that are Llama Chat exports and returns the ones that are not. */
	importExports: (files: File[]) => Promise<File[]>;
}

const DROP_IMPORT_CONTEXT = Symbol('drop-import');

/** Set by the root layout, which owns the import dialogs. */
export function setDropImportContext(context: DropImportContext): void {
	setContext(DROP_IMPORT_CONTEXT, context);
}

/** Undefined outside the app layout (e.g. in a story), where drops are only attached. */
export function getDropImportContext(): DropImportContext | undefined {
	return getContext<DropImportContext | undefined>(DROP_IMPORT_CONTEXT);
}
