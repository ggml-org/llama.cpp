// Guards the draft badges a repo row shows: a loaded model records its draft in
// the `--model-draft` args, which know the file's quant, and the listing reports
// the kinds of every cached model, which do not. The row shows each kind once.

import { ModelAuxSidecar, ModelDraftSidecar, ServerModelStatus } from '$lib/enums';
import { modelsStore } from '$lib/stores';
import type { ApiModelDataEntry } from '$lib/types/api';
import { beforeEach, describe, expect, it } from 'vitest';

const REPO = 'ggml-org/Qwen3.8-Flash-Next-GGUF';

/** A router entry; `args` stand in for the args of a loaded instance. */
function entry(
	id: string,
	sidecars: string[] = [],
	args: string[] = [],
	status = ServerModelStatus.UNLOADED
) {
	return {
		created: 0,
		id,
		in_cache: true,
		object: 'model',
		owned_by: 'llamacpp',
		path: `/models/${id}`,
		sidecars,
		status: { args, value: status }
	} as unknown as ApiModelDataEntry;
}

describe('draft sidecar badges', () => {
	beforeEach(() => {
		modelsStore.routerModels = [];
	});

	it('shows one badge for a draft the args and the listing both report', () => {
		modelsStore.routerModels = [
			entry(
				`${REPO}:IQ4_NL`,
				[],
				[
					'--model-draft',
					`/cache/models--ggml-org--Qwen3.8-Flash-Next-GGUF/snapshots/${'a'.repeat(40)}/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`
				]
			),
			entry(`${REPO}:Q8_0`, [ModelAuxSidecar.MMPROJ, ModelDraftSidecar.MTP])
		];

		// the args badge wins: it knows the quant the listing does not
		expect(modelsStore.status.getDraftSidecars(REPO)).toEqual([
			{ kind: ModelDraftSidecar.MTP, quant: 'Q8_0', repo: REPO }
		]);
	});

	it('shows one badge for a kind two cached quants report', () => {
		modelsStore.routerModels = [
			entry(`${REPO}:IQ4_NL`, [ModelDraftSidecar.MTP]),
			entry(`${REPO}:Q8_0`, [ModelDraftSidecar.MTP])
		];

		expect(modelsStore.status.getDraftSidecars(REPO)).toEqual([
			{ kind: ModelDraftSidecar.MTP, quant: null, repo: REPO }
		]);
	});

	it('marks the drafts of a download on its way as pending', () => {
		modelsStore.routerModels = [
			entry(
				`${REPO}:Q4_0`,
				[ModelAuxSidecar.MMPROJ, ModelDraftSidecar.MTP],
				[],
				ServerModelStatus.DOWNLOADING
			)
		];

		// the kinds are not on disk yet, so the row reads them as in flight
		expect(modelsStore.status.getDraftSidecars(REPO)).toEqual([
			{ kind: ModelDraftSidecar.MTP, pending: true, quant: null, repo: REPO }
		]);
	});

	it('keeps the on-disk badge of a kind a download also pulls', () => {
		modelsStore.routerModels = [
			entry(`${REPO}:IQ4_NL`, [ModelDraftSidecar.MTP], [], ServerModelStatus.INCOMPLETE),
			entry(
				`${REPO}:Q8_0`,
				[],
				[
					'--model-draft',
					`/cache/models--ggml-org--Qwen3.8-Flash-Next-GGUF/snapshots/${'a'.repeat(40)}/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`
				]
			)
		];

		expect(modelsStore.status.getDraftSidecars(REPO)).toEqual([
			{ kind: ModelDraftSidecar.MTP, quant: 'Q8_0', repo: REPO }
		]);
	});
});
