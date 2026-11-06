import type { ModelSidecar } from '$lib/constants';
import type { ModelDraftSidecar, ModelSelectableFileKind } from '$lib/enums';
import type { HfModelSibling } from '$lib/types/huggingface';

/**
 * Option of a quant `<select>` in the download-options command builder; the
 * picks only compose the serve command and are not bound to the quant chips.
 */
export interface ModelQuantOption {
	/** Quant token, or the file name when the file carries no quant (e.g. BF16). */
	label: string;
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

/**
 * Draft file joined to a main quant: the server pulls the two together, so a
 * chip carries its draft as a badge instead of a download of its own.
 */
export interface ModelCompanionDraft {
	/** Draft sidecar kind, e.g. `mtp`. */
	kind: ModelDraftSidecar;
	/** True when the listing reports the kind already on disk. */
	onDisk: boolean;
	/** Quant of the draft file itself; null when its name carries none. */
	quant: string | null;
	path: string;
	size: number;
}

/** A main quant chip: its file, its download state and the draft that rides with it. */
export interface ModelQuantChip extends ModelSelectableFile {
	/** Draft the chip pulls with the model, badged on the chip; null when none of its kind matches. */
	companion: ModelCompanionDraft | null;
	/** Quant of the chip's own file; null when its name carries none. */
	quant: string | null;
	/** Sidecar kinds the download posts on top of the model's own; the joined draft. */
	sidecars: ModelSidecar[];
	state: ModelDownloadEntryState;
}
