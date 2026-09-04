import { isAuxSidecar, isDraftSidecar, type ModelSidecar } from '$lib/constants';
import { ModelAuxSidecar, ModelDraftSidecar } from '$lib/enums';
import { HuggingFaceService } from '$lib/services';
import type { HfModelSibling } from '$lib/types/huggingface';

/**
 * Option of a quant `<select>`; the picks only compose the serve command and
 * are not bound to the quant chips.
 */
export interface QuantOption {
	/** Quant token, or the file name when the file carries no quant (e.g. BF16). */
	label: string;
	/** Repo-relative file path the option stands for. */
	path: string;
}

/** Download state of a single repo entry, injected by the integration layer. */
export interface DownloadEntryState {
	/** Server identifier the entry's actions (pause / resume / cancel / retry) target. */
	repoWithTag: string;
	isDownloading: boolean;
	progress: ModelDownloadProgress | null;
	isDownloaded: boolean;
	isPaused: boolean;
	isFailed: boolean;
}

/** Download state of a single repo entry, injected by the integration layer. */

export type BitDepthRow = { bitDepth: number; files: HfModelSibling[] };

/** A selectable GGUF, tagged with its kind: main weights, draft, or aux (mmproj). */
export type SelectableFile = HfModelSibling & { kind: 'main' | 'draft' | 'aux' };

/** Kind of a file path: the main weights, a draft sidecar, or an aux sidecar (mmproj). */
export function classify(path: string): 'main' | 'draft' | 'aux' {
	const sidecar = HuggingFaceService.extractQuantMeta(path)?.sidecar;

	if (!sidecar) return 'main';

	return isAuxSidecar(sidecar) ? 'aux' : 'draft';
}

/** Display label of a file: its quant, else the file name without the extension. */
export function labelFor(path: string): string {
	const meta = HuggingFaceService.extractQuantMeta(path);

	if (meta?.quant) return meta.quant;

	// Quantless sidecar files: the chip badge already carries the sidecar type,
	// so the label only marks draft files; aux sidecars badge alone.
	if (meta?.sidecar) return isDraftSidecar(meta.sidecar) ? 'draft' : '';

	const basename = path.split('/').pop() ?? path;

	return basename.replace(/\.gguf$/i, '');
}

/**
 * Bit depth of a quant token; `99` (Other) when it carries none. The `UD-`
 * unsloth prefix is stripped before matching.
 */
export function quantBitDepth(quant: string | null): number {
	if (!quant) return 99;

	const stripped = quant.match(/^UD-(?=.)/i) ? quant.slice(3) : quant;
	const bit = stripped.match(/(?:I?Q|F)(\d+)/i)?.[1];

	return bit ? Number(bit) : 99;
}

// LLAMA-APP-REUSE: --spec-type value for each draft sidecar; aux sidecars stay empty
export const SPEC_TYPE: Record<ModelSidecar, string> = {
	[ModelAuxSidecar.IMATRIX]: '',
	[ModelAuxSidecar.MMPROJ]: '',
	[ModelDraftSidecar.DFLASH]: 'draft-dflash',
	[ModelDraftSidecar.DSPARK]: 'draft-dspark',
	[ModelDraftSidecar.EAGLE3]: 'eagle3',
	[ModelDraftSidecar.MTP]: 'draft-mtp'
};
