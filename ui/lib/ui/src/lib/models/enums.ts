// From ui/apps/chat's src/lib/enums/model.enums.ts: the parts Model UI reads.

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
