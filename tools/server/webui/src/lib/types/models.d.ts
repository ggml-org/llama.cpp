import type { ApiModelDataEntry, ApiModelDetails } from '$lib/types/api';

export interface ModelModalities {
	vision: boolean;
	audio: boolean;
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
}

/**
 * Modality capabilities for file validation
 */
export interface ModalityCapabilities {
	hasVision: boolean;
	hasAudio: boolean;
}
