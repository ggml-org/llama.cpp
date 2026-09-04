// LLAMA-APP-REUSE: model modality / capability / sidecar enums
export enum ModelModality {
	AUDIO = 'AUDIO',
	TEXT = 'TEXT',
	VIDEO = 'VIDEO',
	VISION = 'VISION'
}

export enum ModelCapability {
	REASONING = 'reasoning',
	TOOL_USE = 'tools'
}

/**
 * Speculative-decoding draft sidecars (server spec-type draft-*).
 * Filenames use the lowercase token, e.g. `mtp-<name>.gguf` or `-mtp` suffix.
 */
export enum ModelDraftSidecar {
	/** DFlash block-diffusion draft (spec-type draft-dflash). */
	DFLASH = 'dflash',
	/** DSpark block-diffusion draft (spec-type draft-dspark). */
	DSPARK = 'dspark',
	/** EAGLE-3 speculative draft (spec-type draft-eagle3). */
	EAGLE3 = 'eagle3',
	/** Multi-token-prediction draft head (spec-type draft-mtp). */
	MTP = 'mtp'
}

/**
 * Non-draft sidecar file types. A sidecar is any auxiliary GGUF file
 * accompanying the main model weights.
 */
export enum ModelAuxSidecar {
	/** Importance-matrix data used to build imatrix quants; not loaded at serve time. */
	IMATRIX = 'imatrix',
	/** Multimodal projector: unlocks vision and/or audio input modalities. */
	MMPROJ = 'mmproj'
}

/**
 * Role of a selectable GGUF in the download options: the main weights, a
 * speculative-decoding draft sidecar, or an auxiliary sidecar (mmproj).
 */
export enum SelectableFileKind {
	AUX = 'aux',
	DRAFT = 'draft',
	MAIN = 'main'
}

/**
 * Why an in-flight download is being stopped, so the terminal `download_failed`
 * feed event can be attributed: a user pause (resumable) or a user cancel
 * (discard). Distinguishes these from a genuine download failure.
 */
export enum DownloadStopRequest {
	CANCEL = 'cancel',
	PAUSE = 'pause'
}

/**
 * Destructive download action the user is asked to confirm: stop and discard an
 * in-flight download, or delete an already-downloaded model from disk. Both
 * resolve through the same store removal call, differing only in the copy.
 */
export enum DownloadConfirmAction {
	CANCEL = 'cancel',
	DELETE = 'delete'
}
