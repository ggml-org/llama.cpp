import type { ApiModelDataEntry, ApiModelDetails, ApiModelLoadStage } from '$lib/types/api';

export interface ModelModalities {
	vision: boolean;
	audio: boolean;
	video: boolean;
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

export interface ParsedModelId {
	raw: string;
	orgName: string | null;
	modelName: string | null;
	params: string | null;
	activatedParams: string | null;
	quantization: string | null;
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
