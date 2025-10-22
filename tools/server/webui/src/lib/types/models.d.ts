import type { ApiModelDataEntry, ApiModelDetails } from '$lib/types/api';

export interface ModelOption {
	id: string;
	name: string;
	model: string;
	description?: string;
	capabilities: string[];
	details?: ApiModelDetails['details'];
	meta?: ApiModelDataEntry['meta'];
}
