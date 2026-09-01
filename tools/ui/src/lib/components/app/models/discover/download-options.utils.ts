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

/** Display label of a file: its quant (plus `shared` for shared draft variants), else the file name without the extension. */
export function labelFor(path: string): string {
	const meta = HuggingFaceService.extractQuantMeta(path);

	if (meta?.quant) return meta.shared ? `${meta.quant} shared` : meta.quant;

	const basename = path.split('/').pop() ?? path;

	return basename.replace(/\.gguf$/i, '');
}

// min-w keeps the value clear of the native chevron: Safari sizes a select
// to its widest option, so an exactly-as-wide value would otherwise let the
// chevron overlap the text (draft selects are all same-width quants).
export const SELECT_CLASS =
	'h-7 min-w-18 max-w-40 shrink-0 cursor-pointer rounded-md border border-input bg-transparent py-0 pr-3 pl-2 font-mono text-xs outline-none transition-colors hover:bg-accent/40 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]';
