/**
 * ModelStatusManager - model load/unload operations and the /models/sse feed
 *
 * The feed drives status and progress, replacing any post-operation polling.
 * Created and owned by modelsStore, which also owns the router rows it updates.
 */

import {
	CLI_FLAGS,
	HF_UD_QUANT_PREFIX_REGEX,
	MODEL_ID,
	PATH_SEPARATOR,
	PAUSED_MODEL_DOWNLOADS_LOCALSTORAGE_KEY
} from '$lib/constants';
import { ModelDownloadStopRequest, ServerModelsSseEventType, ServerModelStatus } from '$lib/enums';
import { HuggingFaceService } from '$lib/services/huggingface.service';
import { ModelsService } from '$lib/services/models.service';
import type { ModelPropsManager } from '$lib/stores/models/props.svelte';
// direct imports between stores, not via the barrel, to avoid circular deps
import { serverStore } from '$lib/stores/server.svelte';
// explicit type imports: the app.d.ts globals resolve to `any`, so import the real types
import type {
	ApiModelsSseDownloadProgressData,
	ModelDownloadEntry,
	ModelDownloadProgress
} from '$lib/types';
import type { ModelSidecarBadge } from '$lib/types/models';
import { repoOf } from '$lib/utils/model-names';
import { isAuxSidecar } from '$lib/utils/sidecars';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';

/**
 * The slice of modelsStore the manager drives. Kept narrow on purpose so it
 * cannot reach around the host's full surface; modelsStore implements this
 * structurally.
 */
export interface ModelStatusHost {
	error: string | null;
	readonly props: ModelPropsManager;
	/** Router model rows the status feed updates. */
	routerModels: ApiModelDataEntry[];
	fetchRouterModels(): Promise<void>;
	isModelLoaded(modelId: string): boolean;
	toDisplayName(id: string): string;
}

/**
 * Comparison key of a `<repo>:<tag>` download identifier: uppercased, with the
 * `UD-` quant prefix stripped. The router derives cached model names from the
 * actual file, which drops the prefix, so `repo:UD-Q4_K_XL` and `repo:Q4_K_XL`
 * must compare equal.
 */
function downloadIdKey(repoWithTag: string): string {
	const idx = repoWithTag.indexOf(MODEL_ID.QUANTIZATION_SEPARATOR);
	const repo = idx === -1 ? repoWithTag : repoWithTag.slice(0, idx);
	const tag = idx === -1 ? '' : repoWithTag.slice(idx + 1);

	return `${repo.toUpperCase()}:${tag.toUpperCase().replace(HF_UD_QUANT_PREFIX_REGEX, '')}`;
}

export class ModelStatusManager {
	/**
	 * Sidecar files pulled by registered models, as `<repo>/<file>` keys.
	 * Sidecars are not separate /v1/models entries - the router pulls them as
	 * sidecars of a main model and records them in its `--model-draft` /
	 * `--mmproj` args.
	 */
	private downloadedSidecars = $derived.by(() => {
		const result = new SvelteSet<string>();

		for (const m of this.host.routerModels) {
			const args = m.status?.args;

			if (!args) continue;

			for (let i = 0; i < args.length - 1; i++) {
				if (
					args[i] !== CLI_FLAGS.MODEL_DRAFT &&
					args[i] !== CLI_FLAGS.MODEL_DRAFT_SHORT &&
					args[i] !== CLI_FLAGS.MMPROJ
				) {
					continue;
				}

				const parsed = HuggingFaceService.parseCachePath(args[i + 1]);

				if (parsed) result.add(`${parsed.repo}${PATH_SEPARATOR}${parsed.file}`);
			}
		}

		return result;
	});

	private downloadProgress = new SvelteMap<string, ModelDownloadProgress>();
	/**
	 * Draft sidecars the registered models pull with them, keyed by the repo of the model
	 * that pulls them: the router records them in the model's `--model-draft` args.
	 */
	private draftSidecarsByRepo = $derived.by(() => {
		const result = new SvelteMap<string, ModelSidecarBadge[]>();

		for (const m of this.host.routerModels) {
			const args = m.status?.args;

			if (!args) continue;

			const baseRepo = repoOf(m.id);

			for (let i = 0; i < args.length - 1; i++) {
				if (args[i] !== CLI_FLAGS.MODEL_DRAFT && args[i] !== CLI_FLAGS.MODEL_DRAFT_SHORT) {
					continue;
				}

				const parsed = HuggingFaceService.parseCachePath(args[i + 1]);

				if (!parsed) continue;

				const meta = HuggingFaceService.extractQuantMeta(parsed.file);

				if (!meta?.sidecar || isAuxSidecar(meta.sidecar)) continue;

				const badge: ModelSidecarBadge = {
					kind: meta.sidecar,
					quant: meta.quant,
					repo: parsed.repo
				};
				const kinds = result.get(baseRepo);

				if (kinds) {
					if (!kinds.some((b) => b.kind === badge.kind && b.repo === badge.repo)) {
						kinds.push(badge);
					}
				} else {
					result.set(baseRepo, [badge]);
				}
			}
		}

		return result;
	});
	/** `<repo>:<tag>` strings whose most recent download attempt failed (download_failed). */
	private failedDownloads = new SvelteSet<string>();
	private loadingStates = new SvelteMap<string, boolean>();
	private loadProgress = new SvelteMap<string, ModelLoadProgress>();
	/** Paused downloads with their last reported progress, or null when none arrived before the pause. */
	private pausedDownloads = new SvelteMap<string, ModelDownloadProgress | null>();
	// /models/sse feed state, the single source of truth for status and load progress
	private statusAbort: AbortController | null = null;
	private statusReaderActive = false;
	private statusWaiters = new SvelteMap<
		string,
		{ target: ServerModelStatus; resolve: () => void; reject: (e: Error) => void }
	>();
	/** Tags the user asked to stop (pause or cancel); the download_failed the stop triggers is intentional, not a failure. */
	private stopRequests = new SvelteMap<string, ModelDownloadStopRequest>();

	/**
	 * Cancel an in-flight download or remove a downloaded/failed entry from
	 * the cache (ROUTER mode only).
	 */
	async cancelDownload(repoWithTag: string): Promise<boolean> {
		if (!serverStore.isRouterMode) {
			toast.error('Model downloads are only available in router mode');

			return false;
		}

		this.subscribe();

		// in-flight: the kill triggers download_failed over the feed; mark it as a
		// user cancel so it settles silently instead of toasting a failure
		if (this.downloadProgress.has(repoWithTag)) {
			this.stopRequests.set(repoWithTag, ModelDownloadStopRequest.CANCEL);
		}

		// a downloaded model registers under the name the router derived from the
		// cached file (e.g. the UD- quant prefix is dropped), so resolve the tag to
		// the registered id before asking the server to remove it
		const registeredId =
			this.host.routerModels.find((m) => downloadIdKey(m.id) === downloadIdKey(repoWithTag))?.id ??
			repoWithTag;

		try {
			const res = await ModelsService.cancelDownload(registeredId);
			const ok = res.success === true;

			if (ok) {
				this.downloadProgress.delete(repoWithTag);
				this.failedDownloads.delete(repoWithTag);
				this.deletePausedDownload(repoWithTag);
			}

			return ok;
		} catch (error) {
			toast.error(`Failed to cancel: ${error instanceof Error ? error.message : 'unknown error'}`);

			return false;
		}
	}

	/**
	 * The server force-kills a LOADING model on unload, and the feed reports the
	 * settled status, so no waiter is registered here.
	 */
	async cancelLoad(modelId: string): Promise<void> {
		if (!serverStore.isRouterMode) return;

		this.subscribe();

		try {
			await ModelsService.unload(modelId);
			toast.info(`Load cancelled: ${this.host.toDisplayName(modelId)}`);
		} catch (error) {
			toast.error(`Failed to cancel load: ${this.host.toDisplayName(modelId)}`);

			throw error;
		}
	}

	constructor(private host: ModelStatusHost) {
		// the server has no notion of a paused download, so the ids survive in
		// localStorage; the progress snapshot is stale after a reload and stays null
		try {
			const raw = localStorage.getItem(PAUSED_MODEL_DOWNLOADS_LOCALSTORAGE_KEY);
			const parsed: unknown = JSON.parse(raw ?? '[]');

			if (!Array.isArray(parsed)) return;

			for (const repoWithTag of parsed.filter((id): id is string => typeof id === 'string')) {
				this.pausedDownloads.set(repoWithTag, null);
			}
		} catch {
			// unreadable or corrupt: start without the paused set
		}
	}

	/**
	 * POST /models starts the download in the background; the feed reports
	 * progress, and models_reload refreshes the list once it finishes.
	 * Re-posting a paused tag resumes from the partial files kept on disk.
	 */
	async downloadModel(repoWithTag: string): Promise<void> {
		if (!serverStore.isRouterMode) {
			toast.error('Model downloads are only available in router mode');

			return;
		}

		// the feed must be live so the resulting models_reload event refreshes the list
		this.subscribe();

		// resuming a paused download: drop the paused state, and let the server
		// discard its stale DOWNLOADED entry (via the list fetch) before re-posting
		if (this.deletePausedDownload(repoWithTag) || this.stopRequests.delete(repoWithTag)) {
			await this.host.fetchRouterModels();
		}

		try {
			const res = await ModelsService.downloadModel(repoWithTag);

			if (!res.success) {
				throw new Error(res.error?.message ?? 'Server rejected the download request');
			}

			// flip the chip to "downloading" right away; the feed refines it with real progress
			this.downloadProgress.set(repoWithTag, { downloadedBytes: 0, files: {}, totalBytes: 0 });

			toast.success(`Download started: ${this.host.toDisplayName(repoWithTag)}`);
		} catch (error) {
			toast.error(`Download failed: ${repoWithTag}`);

			throw error;
		}
	}

	async ensureLoaded(modelId: string): Promise<void> {
		if (this.host.isModelLoaded(modelId)) return;

		await this.load(modelId);
	}

	/**
	 * Tracked downloads (in flight or paused) for the selector's
	 * "Download in progress" section.
	 */
	getDownloadEntries(): ModelDownloadEntry[] {
		const inFlight = Array.from(this.downloadProgress, ([repoWithTag, progress]) => ({
			isPaused: false,
			progress,
			repoWithTag
		}));
		const paused = Array.from(this.pausedDownloads, ([repoWithTag, progress]) => ({
			isPaused: true,
			progress,
			repoWithTag
		}));

		return [...inFlight, ...paused];
	}

	getDownloadProgress(repoWithTag: string): ModelDownloadProgress | null {
		return this.downloadProgress.get(repoWithTag) ?? null;
	}

	/** Draft sidecars a repo pulls with its model, e.g. `mtp` at `Q8_0`. */
	getDraftSidecars(repoId: string): ModelSidecarBadge[] {
		return this.draftSidecarsByRepo.get(repoId) ?? [];
	}

	getLoadProgress(modelId: string): ModelLoadProgress | null {
		return this.loadProgress.get(modelId) ?? null;
	}

	getPausedDownloadProgress(repoWithTag: string): ModelDownloadProgress | null {
		return this.pausedDownloads.get(repoWithTag) ?? null;
	}

	hasFailedDownload(repoWithTag: string): boolean {
		return this.failedDownloads.has(repoWithTag);
	}

	/** Active while the feed reports download_progress for the tag. */
	isDownloadInProgress(repoWithTag: string): boolean {
		return this.downloadProgress.has(repoWithTag);
	}

	isDownloadPaused(repoWithTag: string): boolean {
		return this.pausedDownloads.has(repoWithTag);
	}

	/**
	 * True when the tag is already registered in the /v1/models list; both ids
	 * are normalized, see downloadIdKey().
	 */
	isModelDownloaded(repoWithTag: string): boolean {
		const key = downloadIdKey(repoWithTag);

		return this.host.routerModels.some((m) => downloadIdKey(m.id) === key);
	}

	isOperationInProgress(modelId: string): boolean {
		return this.loadingStates.get(modelId) ?? false;
	}

	isSidecarDownloaded(repoId: string, filePath: string): boolean {
		return this.downloadedSidecars.has(`${repoId}/${filePath}`);
	}

	async load(modelId: string): Promise<void> {
		if (this.host.isModelLoaded(modelId)) return;

		if (this.loadingStates.get(modelId)) return;

		this.loadingStates.set(modelId, true);
		this.host.error = null;

		// the feed drives completion, so it must be live before the request
		this.subscribe();

		const reachedLoaded = this.waitForStatus(modelId, ServerModelStatus.LOADED);

		reachedLoaded.catch(() => {});

		try {
			await ModelsService.load(modelId);
			await reachedLoaded;
			toast.success(`Model loaded: ${this.host.toDisplayName(modelId)}`);
		} catch (error) {
			this.rejectStatus(modelId, error instanceof Error ? error : new Error('load failed'));
			this.host.error = error instanceof Error ? error.message : 'Failed to load model';
			toast.error(`Failed to load model: ${this.host.toDisplayName(modelId)}`);

			throw error;
		} finally {
			this.loadingStates.set(modelId, false);
		}
	}

	/**
	 * The server stops the download child but keeps the partial files, so
	 * re-posting the tag resumes where it stopped. The feed reports the stop
	 * as download_failed; the 'pause' stop request marks it as intentional.
	 */
	async pauseDownload(repoWithTag: string): Promise<void> {
		if (!serverStore.isRouterMode) {
			toast.error('Model downloads are only available in router mode');

			return;
		}

		this.subscribe();

		this.stopRequests.set(repoWithTag, ModelDownloadStopRequest.PAUSE);

		try {
			await ModelsService.unload(repoWithTag);
		} catch {
			this.stopRequests.delete(repoWithTag);
			toast.error(`Failed to pause: ${repoWithTag}`);
		}
	}

	/** Open the /models/sse feed with auto reconnect; idempotent, router mode only. */
	subscribe(): void {
		if (this.statusReaderActive) return;

		if (!serverStore.isRouterMode) return;

		this.statusReaderActive = true;
		this.statusAbort = new AbortController();
		void this.runStatusReader(this.statusAbort.signal);
	}

	async unload(modelId: string): Promise<void> {
		if (!this.host.isModelLoaded(modelId)) return;

		if (this.loadingStates.get(modelId)) return;

		this.loadingStates.set(modelId, true);
		this.host.error = null;

		this.subscribe();

		const reachedUnloaded = this.waitForStatus(modelId, ServerModelStatus.UNLOADED);

		reachedUnloaded.catch(() => {});

		try {
			await ModelsService.unload(modelId);
			await reachedUnloaded;
			toast.info(`Model unloaded: ${this.host.toDisplayName(modelId)}`);
		} catch (error) {
			this.rejectStatus(modelId, error instanceof Error ? error : new Error('unload failed'));
			this.host.error = error instanceof Error ? error.message : 'Failed to unload model';
			toast.error(`Failed to unload model: ${this.host.toDisplayName(modelId)}`);

			throw error;
		} finally {
			this.loadingStates.set(modelId, false);
		}
	}

	unsubscribe(): void {
		this.statusReaderActive = false;
		this.statusAbort?.abort();
		this.statusAbort = null;
		this.loadProgress.clear();
		this.downloadProgress.clear();
		this.failedDownloads.clear();
		this.stopRequests.clear();
	}

	/**
	 * A user pause keeps the last progress and stays resumable, a user cancel
	 * settles silently; genuine failures are marked so the UI can offer a retry.
	 */
	private applyDownloadFinished(event: ApiModelsSseEvent): void {
		let request: ModelDownloadStopRequest | undefined;

		if (event.event === ServerModelsSseEventType.DOWNLOAD_FAILED) {
			request = this.stopRequests.get(event.model);
			this.stopRequests.delete(event.model);
		}

		const progress = this.downloadProgress.get(event.model) ?? null;

		this.downloadProgress.delete(event.model);

		if (request === ModelDownloadStopRequest.CANCEL) {
			// user cancel: settle silently, the feed's model_remove cleans up the entry
			this.failedDownloads.delete(event.model);
			this.deletePausedDownload(event.model);

			return;
		}

		if (request === ModelDownloadStopRequest.PAUSE) {
			this.setPausedDownload(event.model, progress);
			this.failedDownloads.delete(event.model);

			return;
		}

		this.deletePausedDownload(event.model);

		const ok = event.event === ServerModelsSseEventType.DOWNLOAD_FINISHED;

		if (ok) {
			this.failedDownloads.delete(event.model);

			// the finished download only registers in /v1/models on the next list
			// fetch (the server reloads its model table then), so refetch to flip
			// the quant chips to "downloaded" without waiting for a dialog reopen
			void this.host.fetchRouterModels();

			toast.success(`Download finished: ${this.host.toDisplayName(event.model)}`);
		} else {
			this.failedDownloads.add(event.model);
			toast.error(`Download failed: ${this.host.toDisplayName(event.model)}`);
		}
	}

	/** Aggregate per-file progress into downloaded/total byte counts. */
	private applyDownloadProgress(event: ApiModelsSseEvent): void {
		const data = event.data;

		if (!data || !('progress' in data)) return;

		const progress = (data as ApiModelsSseDownloadProgressData).progress;

		let downloaded = 0;
		let total = 0;

		for (const file of Object.values(progress)) {
			downloaded += file?.done ?? 0;
			total += file?.total ?? 0;
		}

		this.downloadProgress.set(event.model, {
			downloadedBytes: downloaded,
			files: progress,
			totalBytes: total
		});
	}

	private applyModelStatus(event: ApiModelsSseEvent): void {
		const model = event.model;
		const data = event.data;

		if (!model || !data || !('status' in data) || !data.status) return;

		const status = data.status;

		this.setRouterModelStatus(model, status);

		if (status === ServerModelStatus.LOADING) {
			if (data.progress) this.loadProgress.set(model, data.progress);
		} else {
			this.loadProgress.delete(model);
		}

		if (status === ServerModelStatus.LOADED) {
			void this.host.props.updateModelModalities(model);
		}

		const failed =
			status === ServerModelStatus.FAILED ||
			(status === ServerModelStatus.UNLOADED && (data.exit_code ?? 0) !== 0);

		if (failed) {
			this.rejectStatus(model, new Error(`Model failed: ${this.host.toDisplayName(model)}`));

			return;
		}

		this.settleStatus(model, status);
	}

	/** Route one feed record by event kind. */
	private applyStatusEvent(event: ApiModelsSseEvent): void {
		switch (event.event) {
			case ServerModelsSseEventType.STATUS_CHANGE:
			case ServerModelsSseEventType.MODEL_STATUS:
			case ServerModelsSseEventType.STATUS_UPDATE:
				this.applyModelStatus(event);

				break;
			case ServerModelsSseEventType.MODELS_RELOAD:
				void this.host.fetchRouterModels();

				break;
			case ServerModelsSseEventType.MODEL_REMOVE:
				this.removeRouterModel(event.model);

				break;
			case ServerModelsSseEventType.DOWNLOAD_PROGRESS:
				this.applyDownloadProgress(event);

				break;
			case ServerModelsSseEventType.DOWNLOAD_FINISHED:
			case ServerModelsSseEventType.DOWNLOAD_FAILED:
				this.applyDownloadFinished(event);

				break;
		}
	}

	private deletePausedDownload(repoWithTag: string): boolean {
		if (!this.pausedDownloads.delete(repoWithTag)) return false;

		this.persistPausedDownloads();

		return true;
	}

	private persistPausedDownloads(): void {
		try {
			localStorage.setItem(
				PAUSED_MODEL_DOWNLOADS_LOCALSTORAGE_KEY,
				JSON.stringify(Array.from(this.pausedDownloads.keys()))
			);
		} catch {
			// storage unavailable: the pauses just do not survive a reload
		}
	}

	private rejectStatus(modelId: string, error: Error): void {
		const waiter = this.statusWaiters.get(modelId);

		if (waiter) {
			this.statusWaiters.delete(modelId);
			waiter.reject(error);
		}
	}

	private removeRouterModel(modelId: string): void {
		if (this.host.routerModels.findIndex((m) => m.id === modelId) === -1) return;

		this.host.routerModels = this.host.routerModels.filter((m) => m.id !== modelId);
		this.loadProgress.delete(modelId);
		this.downloadProgress.delete(modelId);
		this.failedDownloads.delete(modelId);
		this.deletePausedDownload(modelId);
		this.stopRequests.delete(modelId);
		this.rejectStatus(modelId, new Error(`Model removed: ${this.host.toDisplayName(modelId)}`));

		// drop the row from the selector options too; they rebuild from the list
		// response, which only a refetch provides
		void this.host.fetchRouterModels();
	}

	private async runStatusReader(signal: AbortSignal): Promise<void> {
		await ModelsService.watchModelEvents(signal, (event) => this.applyStatusEvent(event));
	}

	private setPausedDownload(repoWithTag: string, progress: ModelDownloadProgress | null): void {
		this.pausedDownloads.set(repoWithTag, progress);
		this.persistPausedDownloads();
	}

	// reassign the array: mutating an entry in place would not trigger reactivity
	private setRouterModelStatus(modelId: string, status: ServerModelStatus): void {
		const idx = this.host.routerModels.findIndex((m) => m.id === modelId);

		if (idx === -1) return;

		const current = this.host.routerModels[idx];

		if (current.status.value === status) return;

		const next = [...this.host.routerModels];

		next[idx] = { ...current, status: { ...current.status, value: status } };
		this.host.routerModels = next;
	}

	private settleStatus(modelId: string, status: ServerModelStatus): void {
		const waiter = this.statusWaiters.get(modelId);

		if (waiter && waiter.target === status) {
			this.statusWaiters.delete(modelId);
			waiter.resolve();
		}
	}

	// one operation runs per model at a time, so one waiter per model suffices
	private waitForStatus(modelId: string, target: ServerModelStatus): Promise<void> {
		return new Promise((resolve, reject) => {
			this.statusWaiters.set(modelId, { reject, resolve, target });
		});
	}
}
