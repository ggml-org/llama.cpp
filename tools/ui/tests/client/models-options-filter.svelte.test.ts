// Guards the model options filter: models with embedded draft heads (e.g. -mtp)
// remain selectable in the UI, while non-loadable auxiliary files (projectors, imatrix)
// and standalone draft-only files are filtered out.

import { ModelDraftSidecar, ServerModelStatus } from '$lib/enums';
import { modelsStore } from '$lib/stores';
import type { ApiModelDataEntry, ApiModelsListResponse } from '$lib/types/api';
import { describe, expect, it } from 'vitest';

function modelEntry(id: string, status = ServerModelStatus.LOADED): ApiModelDataEntry {
	return {
		created: 0,
		id,
		in_cache: true,
		object: 'model',
		owned_by: 'llamacpp',
		path: `/models/${id}`,
		status: { value: status }
	} as unknown as ApiModelDataEntry;
}

describe('buildModelOptions', () => {
	it('preserves loadable models with embedded draft suffix while filtering auxiliary and standalone sidecars', () => {
		const response: ApiModelsListResponse = {
			data: [
				modelEntry('Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf'),
				modelEntry('llama-3.2-1b-instruct-q4_k_m.gguf'),
				modelEntry('mmproj-model-f16.gguf'),
				modelEntry('imatrix.gguf'),
				modelEntry('mtp-Q4_0.gguf'),
				modelEntry('Model-mtp-draft.gguf'),
				modelEntry('downloading-model-q4.gguf', ServerModelStatus.DOWNLOADING)
			],
			object: 'list'
		};
		const store = modelsStore as unknown as {
			buildModelOptions: (res: ApiModelsListResponse) => Array<{
				id: string;
				parsedId: { sidecar: string | null };
			}>;
		};
		const options = store.buildModelOptions(response);
		const optionIds = options.map((opt) => opt.id);

		expect(optionIds).toEqual([
			'Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf',
			'llama-3.2-1b-instruct-q4_k_m.gguf'
		]);

		const mtpOption = options.find(
			(opt) => opt.id === 'Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.gguf'
		);

		expect(mtpOption?.parsedId.sidecar).toBe(ModelDraftSidecar.MTP);
	});
});
