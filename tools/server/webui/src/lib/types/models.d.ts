import type { ApiModelDataEntry, ApiModelDetails } from '$lib/types/api';

/**
 * Model modalities - vision and audio capabilities
 */
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
	/** Model modalities from /props endpoint */
	modalities?: ModelModalities;
	details?: ApiModelDetails['details'];
	meta?: ApiModelDataEntry['meta'];
}
