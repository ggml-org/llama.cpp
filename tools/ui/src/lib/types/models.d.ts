import type { ModelSidecar } from '$lib/constants/model-id.constants';
import type { ApiModelDataEntry, ApiModelDetails, ApiModelLoadStage } from '$lib/types/api';

export interface ModelModalities {
	vision: boolean;
	audio: boolean;
	video: boolean;
}

export interface ModelCapabilities {
	reasoning: boolean;
	tools: boolean;
}

export interface ModelOption {
	id: string;
	name: string;
	model: string;
	description?: string;
	capabilities: string[];
	modalities?: ModelModalities;
	details?: ApiModelDetails['details'];
	meta?: ApiModelDataEntry['meta'];
	parsedId?: ParsedModelId;
	aliases?: string[];
	tags?: string[];
}

/**
 * Ephemeral UI-only load progress for one model instance.
 * Lives only while a load runs, driven by the /models/sse feed.
 * stage is absent until the feed reports its first stage.
 */
export interface ModelLoadProgress {
	stages: ApiModelLoadStage[];
	current: ApiModelLoadStage;
	value: number;
}
/**
 * Per-byte download progress for one in-flight model download, driven by the
 * /models/sse feed. Lives only while a download runs.
 */
export interface ModelDownloadFileProgress {
	/** Bytes downloaded for this file so far. */
	done: number;
	/** Total bytes of the file. */
	total: number;
}

export interface ModelDownloadProgress {
	/** Summed progress across all files of the download plan. */
	downloadedBytes: number;
	/** Summed plan size across all files. */
	totalBytes: number;
	/** Per-file progress keyed by file URL, as reported by the feed. */
	files: Record<string, ModelDownloadFileProgress>;
}

// LLAMA-APP-REUSE: parsed model id shape
export interface ParsedModelId {
	raw: string;
	orgName: string | null;
	modelName: string | null;
	params: string | null;
	activatedParams: string | null;
	quantization: string | null;
	sidecar: ModelSidecar | null;
	tags: string[];
}

/**
 * Modality capabilities for file validation
 */
export interface ModalityCapabilities {
	hasVision: boolean;
	hasAudio: boolean;
	hasVideo: boolean;
}
