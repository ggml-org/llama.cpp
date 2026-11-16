// Guards the download flows the chips depend on: one POST per click, the chip
// reading downloaded once the list reports the model, a resume keeping the
// sidecars it started with, a cancel dropping the whole download, and a reload
// restoring the paused state of an unfinished one.

import { ModelAuxSidecar, ModelDraftSidecar, ServerModelStatus, ServerRole } from '$lib/enums';
import { ModelsService } from '$lib/services/models.service';
import { modelsStore } from '$lib/stores';
import { serverStore } from '$lib/stores/server.svelte';
import type { ApiModelDataEntry, ApiModelsListResponse } from '$lib/types/api';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const REPO = 'org/model-GGUF';
const TAG = `${REPO}:Q4_K_M`;
const DRAFTS = [ModelAuxSidecar.MMPROJ, ModelDraftSidecar.DFLASH];

/** The listing entry a router reports once a tag is downloaded. */
function entry(id: string, sidecars: string[] = [], status = ServerModelStatus.DOWNLOADED) {
	return {
		created: 0,
		id,
		in_cache: true,
		object: 'model',
		owned_by: 'llamacpp',
		path: `/models/${id}`,
		sidecars,
		status: { value: status }
	} as unknown as ApiModelDataEntry;
}

/** The download endpoints the store talks to; the feed is not live in these tests. */
function stubEndpoints() {
	const post = vi.spyOn(ModelsService, 'downloadModel').mockResolvedValue({ success: true });
	const cancel = vi.spyOn(ModelsService, 'cancelDownload').mockResolvedValue({ success: true });
	const unload = vi.spyOn(ModelsService, 'unload').mockResolvedValue({ success: true });

	vi.spyOn(ModelsService, 'watchModelEvents').mockResolvedValue(undefined);
	vi.spyOn(ModelsService, 'list').mockResolvedValue({
		data: [],
		models: [],
		object: 'list'
	} as unknown as ApiModelsListResponse);

	return { cancel, post, unload };
}

describe('model download flows', () => {
	beforeEach(() => {
		vi.restoreAllMocks();
		serverStore.role = ServerRole.ROUTER;
		modelsStore.routerModels = [];
		// the store is a singleton: drop what an earlier case tracked
		modelsStore.status.unsubscribe();
	});

	it('posts once for two quick clicks on the same chip', async () => {
		const { post } = stubEndpoints();
		const first = modelsStore.status.downloadModel(TAG, [ModelDraftSidecar.DFLASH]);
		const second = modelsStore.status.downloadModel(TAG, [ModelDraftSidecar.DFLASH]);

		// the chip reads in flight before the server answers, so the second click
		// finds it busy instead of posting again
		expect(modelsStore.status.isDownloadInProgress(TAG)).toBe(true);

		await Promise.all([first, second]);

		expect(post).toHaveBeenCalledTimes(1);
		expect(post).toHaveBeenCalledWith(TAG, DRAFTS);
	});

	it('reads the chip as downloaded once the list reports the model', async () => {
		stubEndpoints();

		await modelsStore.status.downloadModel(TAG, [ModelDraftSidecar.DFLASH]);

		expect(modelsStore.status.isModelDownloaded(TAG)).toBe(false);

		// the feed's reload lands as a listing the tag now appears in
		modelsStore.routerModels = [entry(TAG, DRAFTS)];

		expect(modelsStore.status.isModelDownloaded(TAG)).toBe(true);
		// the sidecar option of that model reads the draft off the same entry
		expect(modelsStore.status.isModelDownloaded(`${TAG}-dflash`)).toBe(true);
	});

	it('does not pause a download whose POST is still on its way', async () => {
		const { post, unload } = stubEndpoints();

		let release: () => void = () => {};

		post.mockImplementation(
			() =>
				new Promise((resolve) => {
					release = () => resolve({ success: true });
				})
		);

		const started = modelsStore.status.downloadModel(TAG, [ModelDraftSidecar.DFLASH]);

		// the chip reads as downloading before the server answers, so a second click
		// must not stop a model that never started
		await modelsStore.status.pauseDownload(TAG);

		expect(unload).not.toHaveBeenCalled();
		expect(modelsStore.status.isDownloadInProgress(TAG)).toBe(true);
		expect(modelsStore.status.isDownloadPaused(TAG)).toBe(false);

		release();
		await started;
	});

	it('resumes with the sidecars the download started with', async () => {
		const { post } = stubEndpoints();

		await modelsStore.status.downloadModel(TAG, [ModelDraftSidecar.DFLASH]);
		await modelsStore.status.pauseDownload(TAG);

		expect(modelsStore.status.isDownloadPaused(TAG)).toBe(true);

		// the resume a chip sends names no kinds of its own
		await modelsStore.status.downloadModel(TAG);

		expect(post).toHaveBeenLastCalledWith(TAG, DRAFTS);
		expect(modelsStore.status.isDownloadPaused(TAG)).toBe(false);
	});

	it('cancels the model and its sidecars as one download', async () => {
		const { cancel } = stubEndpoints();

		await modelsStore.status.downloadModel(TAG, [ModelDraftSidecar.DFLASH]);
		await modelsStore.status.cancelDownload(TAG);

		// no sidecar of its own is asked for: the model goes as one unit
		expect(cancel).toHaveBeenCalledWith(TAG, null);
		expect(modelsStore.status.isDownloadInProgress(TAG)).toBe(false);
		expect(modelsStore.status.getDownloadEntries()).toEqual([]);
	});

	it('restores the paused state of an unfinished download after a reload', async () => {
		const { post } = stubEndpoints();

		modelsStore.routerModels = [entry(TAG, DRAFTS, ServerModelStatus.INCOMPLETE)];
		modelsStore.status.syncPausedDownloads();

		expect(modelsStore.status.isDownloadPaused(TAG)).toBe(true);
		expect(modelsStore.status.getDownloadEntries()).toEqual([
			{ isPaused: true, progress: null, repoWithTag: TAG }
		]);

		// and its resume keeps the kinds the incomplete listing reports
		await modelsStore.status.downloadModel(TAG);

		expect(post).toHaveBeenCalledWith(TAG, DRAFTS);
	});
});
