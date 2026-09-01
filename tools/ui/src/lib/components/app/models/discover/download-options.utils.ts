import { isAuxSidecar } from '$lib/constants';
import { HuggingFaceService } from '$lib/services';
import type { HfModelSibling } from '$lib/types/huggingface';

/** Download state of a single repo entry, injected by the integration layer. */
export interface DownloadEntryState {
	isDownloading: boolean;
	progress: ModelDownloadProgress | null;
	isDownloaded: boolean;
	isFailed: boolean;
}

export type BitDepthRow = { bitDepth: number; files: HfModelSibling[] };

/** A selectable GGUF, tagged with its kind: main weights, draft, or aux (mmproj). */
export type SelectableFile = HfModelSibling & { kind: 'main' | 'draft' | 'aux' };

/** Option of a quant `<select>`; already-downloaded files stay non-selectable. */
export interface QuantOption {
	disabled: boolean;
	/** Quant token, or the file name when the file carries no quant (e.g. BF16). */
	label: string;
	path: string;
	size: number;
}

/** Kind of a file path: the main weights, a draft sidecar, or an aux sidecar (mmproj). */
export function classify(path: string): 'main' | 'draft' | 'aux' {
	const sidecar = HuggingFaceService.extractQuantMeta(path)?.sidecar;

	if (!sidecar) return 'main';

	return isAuxSidecar(sidecar) ? 'aux' : 'draft';
}

/** Display label of a file: its quant, else the file name without the extension. */
export function labelFor(path: string): string {
	const quant = HuggingFaceService.extractQuantMeta(path)?.quant;

	if (quant) return quant;

	const basename = path.split('/').pop() ?? path;

	return basename.replace(/\.gguf$/i, '');
}
