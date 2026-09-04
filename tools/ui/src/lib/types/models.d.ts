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

/** UI-only load progress for one model, driven by the /models/sse feed. */
export interface ModelLoadProgress {
	stages: ApiModelLoadStage[];
	current: ApiModelLoadStage;
	value: number;
}

/** Per-file bytes of an in-flight download. */
export interface ModelDownloadFileProgress {
	done: number;
	total: number;
}

/** Progress of an in-flight download, summed across its files. */
export interface ModelDownloadProgress {
	downloadedBytes: number;
	totalBytes: number;
	/** Per-file progress keyed by file URL. */
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

/** Modality capabilities for file validation. */
export interface ModalityCapabilities {
	hasVision: boolean;
	hasAudio: boolean;
	hasVideo: boolean;
}
