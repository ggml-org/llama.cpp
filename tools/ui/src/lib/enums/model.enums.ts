export enum ModelModality {
	AUDIO = 'AUDIO',
	TEXT = 'TEXT',
	VIDEO = 'VIDEO',
	VISION = 'VISION'
}

export enum ModelCapability {
	REASONING = 'REASONING',
	TOOL_USE = 'TOOL_USE'
}

/**
 * Speculative-decoding draft sidecars (server spec-type draft-*).
 * Filenames use the lowercase token, e.g. `mtp-<name>.gguf` or `-mtp` suffix.
 */
export enum ModelDraftSidecar {
	/** DFlash block-diffusion draft (spec-type draft-dflash). */
	DFLASH = 'DFLASH',
	/** DSpark block-diffusion draft (spec-type draft-dspark). */
	DSPARK = 'DSPARK',
	/** EAGLE-3 speculative draft (spec-type draft-eagle3). */
	EAGLE3 = 'EAGLE3',
	/** Multi-token-prediction draft head (spec-type draft-mtp). */
	MTP = 'MTP'
}

/**
 * Non-draft sidecar file types. A sidecar is any auxiliary GGUF file
 * accompanying the main model weights.
 */
export enum ModelAuxSidecar {
	/** Multimodal projector: unlocks vision and/or audio input modalities. */
	MMPROJ = 'MMPROJ'
}
