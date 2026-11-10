// The data Model UI takes in. Fetching it (from llama-server, the Hugging Face API or a curated catalog) stays in each
// app; the components only draw it and report what the user did.

export interface ModelModalities {
	vision: boolean;
	audio: boolean;
	video: boolean;
}

/** A min-max size in bytes, across a repo's quants. */
export interface ModelSizeRange {
	min: number;
	max: number;
}

/** Where a quant file stands on this device, as the app knows it. */
export type ModelQuantState = 'idle' | 'downloading' | 'paused' | 'downloaded' | 'failed';

/** A draft model that rides along with a quant, e.g. an MTP head. */
export interface ModelQuantCompanion {
	kind: string;
	quant?: string;
	sizeBytes?: number;
	onDisk?: boolean;
}

export interface ModelDownloadProgress {
	downloadedBytes: number;
	totalBytes: number;
}
