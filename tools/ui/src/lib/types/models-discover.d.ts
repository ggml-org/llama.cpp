import type { ModelSelectableFileKind } from '$lib/enums';
import type { HfModelSibling } from '$lib/types/huggingface';

/**
 * Option of a quant `<select>` in the download-options command builder; the
 * picks only compose the serve command and are not bound to the quant chips.
 */
export interface ModelQuantOption {
	/** Quant token, or the file name when the file carries no quant (e.g. BF16). */
	label: string;
	/** Repo-relative file path the option stands for. */
	path: string;
}

/** Download state of a single repo entry, injected by the integration layer. */
export interface ModelDownloadEntryState {
	/** Server identifier the entry's actions (pause / resume / cancel / retry) target. */
	repoWithTag: string;
	isDownloading: boolean;
	progress: ModelDownloadProgress | null;
	isDownloaded: boolean;
	isPaused: boolean;
	isFailed: boolean;
}

/** A group of GGUF files sharing one bit-depth bucket. */
export type ModelBitDepthRow = { bitDepth: number; files: HfModelSibling[] };

/** A selectable GGUF, tagged with its role: main weights, draft, or aux (mmproj). */
export type ModelSelectableFile = HfModelSibling & { kind: ModelSelectableFileKind };
