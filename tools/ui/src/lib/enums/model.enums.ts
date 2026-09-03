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
	/** Multimodal projector: unlocks vision and/or audio input modalities. */
	MMPROJ = 'mmproj',
	/** Importance-matrix data used to build imatrix quants; not loaded at serve time. */
	IMATRIX = 'imatrix'
}
