/**
 * ModelStatusManager - model load/unload operations and the /models/sse feed
 *
 * The feed drives status and progress, replacing any post-operation polling.
 * Created and owned by modelsStore, which also owns the router rows it updates.
 */

import { browser } from '$app/environment';
import {
	CLI_FLAGS,
	HF_UD_QUANT_PREFIX_REGEX,
	MODEL_ID,
	type ModelSidecar,
	STORAGE_APP_NAME
} from '$lib/constants';
import {
	ModelAuxSidecar,
	ModelDownloadStopRequest,
	ServerModelsSseEventType,
	ServerModelStatus
} from '$lib/enums';
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
import { isAuxSidecar, isDraftSidecar, sidecarFromFileToken } from '$lib/utils/sidecars';
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

/** Sidecar kinds the server pulls with a model, as -hf does. */
const SERVER_SIDECARS: readonly string[] = ['mmproj', 'mtp', 'dflash', 'eagle3', 'dspark'];

/**
 * The model a download option stands for, and the sidecar it adds: `<repo>:<quant>`
 * is the model itself, `<repo>:<quant>-<sidecar>` a sidecar of that model, and a
 * sidecar without quant belongs to the repo's default model.
 */
function downloadTarget(repoWithTag: string): { model: string; sidecar: ModelSidecar | null } {
	const parsed = ModelsService.parseModelId(repoWithTag);

	if (!parsed.sidecar) return { model: repoWithTag, sidecar: null };

	const repo = repoOf(repoWithTag);
	const model = parsed.quantization
		? `${repo}${MODEL_ID.QUANTIZATION_SEPARATOR}${parsed.quantization}`
		: repo;

	return { model, sidecar: parsed.sidecar };
}

/** Zeroed progress, used until the first feed record reports real counts. */
function emptyDownloadProgress(): ModelDownloadProgress {
	return { downloadedBytes: 0, files: {}, totalBytes: 0 };
}

export class ModelStatusManager {
	/** In-flight downloads by model, with their aggregated progress. */
	private downloadProgress = new SvelteMap<string, ModelDownloadProgress>();
	/** Sidecar kinds each in-flight or paused download pulls with its model. */
	private downloadSidecars = new SvelteMap<string, ModelSidecar[]>();
	/**
	 * Draft sidecars the registered models pull with them, keyed by the model that pulls
	 * them: the router records them in the model's `--model-draft` args, and each quant
	 * of a repo can pull a different draft.
	 */
	private draftSidecarsByModel = $derived.by(() => {
		const result = new SvelteMap<string, ModelSidecarBadge[]>();

		for (const m of this.host.routerModels) {
			const args = m.status?.args;

			if (!args) continue;

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
				const kinds = result.get(m.id);

				if (kinds) {
					if (
						!kinds.some(
							(b) => b.kind === badge.kind && b.repo === badge.repo && b.quant === badge.quant
						)
					) {
						kinds.push(badge);
					}
				} else {
					result.set(m.id, [badge]);
				}
			}
		}

		return result;
	});
	/** Models whose most recent download attempt failed (download_failed). */
	private failedDownloads = new SvelteSet<string>();
	/** Draft sidecars the cached models report on disk, keyed by repo. */
	private listedSidecarsByRepo = $derived.by(() =>
		ModelsService.draftSidecarsByRepo(this.host.routerModels)
	);
	private loadingStates = new SvelteMap<string, boolean>();
	private loadProgress = new SvelteMap<string, ModelLoadProgress>();
	/** Paused downloads with their last reported progress, or null when none arrived before the pause. */
	private pausedDownloads = new SvelteMap<string, ModelDownloadProgress | null>();
	/** Models with a download POST in flight, so a second click cannot post again. */
	private pendingDownloads = new SvelteSet<string>();
	/**
	 * Draft kinds a download on its way or unfinished reports, keyed by repo: the
	 * files are not on disk yet, so a download row reads them as pending.
	 */
	private pendingSidecarsByRepo = $derived.by(() => {
		const result = new SvelteMap<string, ModelSidecarBadge[]>();

		for (const m of this.host.routerModels) {
			if (
				m.status?.value !== ServerModelStatus.DOWNLOADING &&
				m.status?.value !== ServerModelStatus.INCOMPLETE
			)
				continue;

			if (!Array.isArray(m.sidecars)) continue;

			const repo = repoOf(m.id);

			for (const kind of m.sidecars) {
				const sidecar = sidecarFromFileToken(kind);

				if (!sidecar || !isDraftSidecar(sidecar)) continue;

				const badge: ModelSidecarBadge = { kind: sidecar, pending: true, quant: null, repo };
				const badges = result.get(repo);

				if (badges) {
					if (!badges.some((b) => b.kind === badge.kind && b.repo === badge.repo)) {
						badges.push(badge);
					}
				} else {
					result.set(repo, [badge]);
				}
			}
		}

		return result;
	});
	// /models/sse feed state, the single source of truth for status and load progress
	private statusAbort: AbortController | null = null;
	private statusReaderActive = false;

	private statusWaiters = new SvelteMap<
		string,
		{ target: ServerModelStatus; resolve: () => void; reject: (e: Error) => void }
	>();
	/** Models the user asked to stop (pause or cancel); the download_failed the stop triggers is intentional, not a failure. */
	private stopRequests = new SvelteMap<string, ModelDownloadStopRequest>();

	/**
	 * Cancel an in-flight download, or remove a downloaded/failed model from the cache
	 * (ROUTER mode only). A sidecar option removes that sidecar alone; anything else
	 * removes the model along with the partial files of its download.
	 */
	async cancelDownload(repoWithTag: string): Promise<boolean> {
		if (!serverStore.isRouterMode) {
			toast.error('Model downloads are only available in router mode');

			return false;
		}

		this.subscribe();

		const { model, sidecar } = downloadTarget(repoWithTag);
		// in-flight: the kill triggers download_failed over the feed; mark it as a
		// user cancel so it settles silently instead of toasting a failure
		const wasInFlight = this.downloadProgress.has(model);

		if (wasInFlight) {
			this.stopRequests.set(model, ModelDownloadStopRequest.CANCEL);
		}

		// a downloaded model registers under the name the router derived from the
		// cached file (e.g. the UD- quant prefix is dropped), so resolve the tag to
		// the registered id before asking the server to remove it
		const registeredId =
			this.host.routerModels.find((m) => downloadIdKey(m.id) === downloadIdKey(model))?.id ?? model;
		const sidecarOnly = sidecar !== null && !wasInFlight && this.isModelDownloaded(model);

		try {
			const res = await ModelsService.cancelDownload(registeredId, sidecarOnly ? sidecar : null);
			const ok = res.success === true;

			if (ok && !sidecarOnly) {
				this.downloadProgress.delete(model);
				this.downloadSidecars.delete(model);
				this.failedDownloads.delete(model);
				this.deletePausedDownload(model);
			}

			// the feed's model_remove refreshes the list too, but refetch here as well:
			// a missed event, or a sidecar removal that sends none, would leave it stale
			if (ok) void this.host.fetchRouterModels();

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
		// the paused-ids registry moved to the server's unfinished-downloads record;
		// drop the key this store used to persist
		if (browser) localStorage.removeItem(`${STORAGE_APP_NAME}.pausedModelDownloads`);
	}

	/**
	 * POST /models starts the download in the background; the feed reports progress and
	 * models_reload refreshes the list once it finishes. A sidecar option posts its model
	 * with that sidecar on top of the ones it already pulls, so the server keeps what is
	 * on disk and adds the rest; re-posting a paused model resumes it.
	 */
	async downloadModel(repoWithTag: string, extraSidecars: ModelSidecar[] = []): Promise<void> {
		if (!serverStore.isRouterMode) {
			toast.error('Model downloads are only available in router mode');

			return;
		}

		const { model, sidecar } = downloadTarget(repoWithTag);

		if (sidecar && !SERVER_SIDECARS.includes(sidecar)) {
			toast.error(`The ${sidecar} file is not downloadable from here`);

			return;
		}

		const registered = this.host.routerModels.find(
			(m) => downloadIdKey(m.id) === downloadIdKey(model)
		);
		const sidecars = [
			ModelAuxSidecar.MMPROJ,
			...(registered?.sidecars ?? []),
			...(this.downloadSidecars.get(model) ?? []),
			...(sidecar ? [sidecar] : []),
			...extraSidecars
		].filter((kind, i, all) => SERVER_SIDECARS.includes(kind) && all.indexOf(kind) === i);

		// a second click must not post again: the server would answer that the model
		// already exists, so a download already on its way is left alone
		if (this.pendingDownloads.has(model) || this.isDownloadInProgress(model)) return;

		this.pendingDownloads.add(model);

		// the feed must be live so the resulting models_reload event refreshes the list
		this.subscribe();

		// the marker and its last reported progress survive until the request is
		// accepted: dropping them first would drop the paused state, which reads as
		// a lost download if the server rejects the resume
		const snapshot = this.pausedDownloads.get(model) ?? null;
		const wasPaused = this.pausedDownloads.has(model);

		// the download reads as in flight from here, so the chip shows it before the
		// server answers; the feed refines it with the real counts
		this.downloadProgress.set(model, snapshot ?? emptyDownloadProgress());

		try {
			const res = await ModelsService.downloadModel(model, sidecars);

			if (!res.success) {
				throw new Error(res.error?.message ?? 'Server rejected the download request');
			}

			// the marker gives way to the live entry, seeded with the last reported
			// progress so the row does not fall back to 0%
			this.deletePausedDownload(model);
			this.stopRequests.delete(model);
			this.failedDownloads.delete(model);
			this.downloadSidecars.set(model, sidecars);

			toast.success(`Download started: ${this.host.toDisplayName(model)}`);
		} catch (error) {
			// the download did not start: the optimistic entry goes, and a resumed one
			// stays paused rather than dropping off the list
			this.downloadProgress.delete(model);

			if (wasPaused) {
				this.setPausedDownload(model, snapshot);
			}

			const reason = error instanceof Error ? error.message : 'unknown error';

			toast.error(`Download failed: ${model} (${reason})`);

			throw error;
		} finally {
			this.pendingDownloads.delete(model);
		}
	}

	async ensureLoaded(modelId: string): Promise<void> {
		if (this.host.isModelLoaded(modelId)) return;

		await this.load(modelId);
	}

	/**
	 * Tracked downloads (in flight or paused) for the selector's "Download in
	 * progress" section: one entry per model, its sidecars riding with it.
	 */
	getDownloadEntries(): ModelDownloadEntry[] {
		const inFlight = Array.from(this.downloadProgress, ([model, progress]) => ({
			isPaused: false,
			progress,
			repoWithTag: model
		}));
		// a model can carry a stale pause mark next to its live entry, e.g. after
		// the list adopted it again while it was resuming: one row, the live one
		const paused = Array.from(this.pausedDownloads, ([model, progress]) => ({
			isPaused: true,
			progress,
			repoWithTag: model
		})).filter((entry) => !this.downloadProgress.has(entry.repoWithTag));

		return [...inFlight, ...paused];
	}

	getDownloadProgress(repoWithTag: string): ModelDownloadProgress | null {
		return this.isDownloadInProgress(repoWithTag)
			? (this.downloadProgress.get(downloadTarget(repoWithTag).model) ?? null)
			: null;
	}

	/**
	 * Draft sidecars a model pulls with it, e.g. `mtp` at `Q8_0`: the router
	 * records each model's sidecars in its `--model-draft` args, and lists the
	 * sidecars each cached repo has on disk.
	 */
	getDraftSidecars(modelId: string): ModelSidecarBadge[] {
		const repo = repoOf(modelId);
		// one badge per kind: the args know the quant of this model's draft, the
		// listing carries repo-wide kinds only, and a draft still coming down is
		// pending rather than on disk
		const merged = [...(this.draftSidecarsByModel.get(modelId) ?? [])];

		for (const source of [this.listedSidecarsByRepo, this.pendingSidecarsByRepo]) {
			for (const badge of source.get(repo) ?? []) {
				if (!merged.some((b) => b.kind === badge.kind && b.repo === badge.repo)) {
					merged.push(badge);
				}
			}
		}

		return merged;
	}

	getLoadProgress(modelId: string): ModelLoadProgress | null {
		return this.loadProgress.get(modelId) ?? null;
	}

	getPausedDownloadProgress(repoWithTag: string): ModelDownloadProgress | null {
		return this.isDownloadPaused(repoWithTag)
			? (this.pausedDownloads.get(downloadTarget(repoWithTag).model) ?? null)
			: null;
	}

	hasFailedDownload(repoWithTag: string): boolean {
		const { model } = downloadTarget(repoWithTag);

		return this.failedDownloads.has(model) && this.pullsSidecar(repoWithTag);
	}

	/** Active while the feed reports download_progress for the option's model, and it pulls the option. */
	isDownloadInProgress(repoWithTag: string): boolean {
		return (
			this.downloadProgress.has(downloadTarget(repoWithTag).model) && this.pullsSidecar(repoWithTag)
		);
	}

	isDownloadPaused(repoWithTag: string): boolean {
		const { model } = downloadTarget(repoWithTag);

		// a running download outranks its pause mark, which can outlive a resume
		return (
			this.pausedDownloads.has(model) &&
			!this.downloadProgress.has(model) &&
			this.pullsSidecar(repoWithTag)
		);
	}

	/**
	 * True when the option's model is registered in the /v1/models list and, for
	 * a sidecar option, reports that sidecar on disk; both ids are normalized,
	 * see downloadIdKey().
	 */
	isModelDownloaded(repoWithTag: string): boolean {
		const { model, sidecar } = downloadTarget(repoWithTag);
		const key = downloadIdKey(model);

		// an unfinished download (downloading or incomplete) is not a model on disk,
		// so the download options offer it as resumable instead
		return this.host.routerModels.some(
			(m) =>
				downloadIdKey(m.id) === key &&
				m.status?.value !== ServerModelStatus.DOWNLOADING &&
				m.status?.value !== ServerModelStatus.INCOMPLETE &&
				(!sidecar || (m.sidecars ?? []).includes(sidecar))
		);
	}

	isOperationInProgress(modelId: string): boolean {
		return this.loadingStates.get(modelId) ?? false;
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
	 * The server stops the download child but keeps the partial files, so re-posting
	 * the model resumes where it stopped. The feed reports the stop as download_failed;
	 * the 'pause' stop request marks it as intentional, and the paused state is stored
	 * before the request: an unconfirmed pause would leave the chip spinning.
	 */
	async pauseDownload(repoWithTag: string): Promise<void> {
		if (!serverStore.isRouterMode) {
			toast.error('Model downloads are only available in router mode');

			return;
		}

		const { model } = downloadTarget(repoWithTag);

		// a POST still on its way has nothing to stop: unloading would kill a model
		// that never started and leave the store neither in flight nor paused
		if (this.pendingDownloads.has(model)) return;

		this.subscribe();

		this.stopRequests.set(model, ModelDownloadStopRequest.PAUSE);

		const snapshot = this.downloadProgress.get(model) ?? null;
		const wasInFlight = this.downloadProgress.delete(model);

		this.setPausedDownload(model, snapshot);

		try {
			await ModelsService.unload(model);
		} catch {
			this.stopRequests.delete(model);
			this.deletePausedDownload(model);

			if (wasInFlight) {
				this.downloadProgress.set(model, snapshot ?? emptyDownloadProgress());
			}

			toast.error(`Failed to pause: ${model}`);
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

	/**
	 * A model the server lists as incomplete is a download that never finished: it
	 * runs in no tab and carries no progress. Offer it as paused with the sidecars
	 * it pulls, and re-posting the model resumes it from the files already on disk.
	 */
	syncPausedDownloads(): void {
		for (const model of this.host.routerModels) {
			if (model.status?.value !== ServerModelStatus.INCOMPLETE) continue;

			if (!this.downloadSidecars.has(model.id)) {
				this.downloadSidecars.set(model.id, model.sidecars ?? []);
			}

			if (this.pausedDownloads.has(model.id)) continue;

			this.setPausedDownload(model.id, null);
		}
	}

	/**
	 * Pause an in-flight download, or resume a paused or incomplete one: one
	 * download per model, its sidecars riding along with it.
	 */
	async toggleDownload(repoWithTag: string): Promise<void> {
		const { model } = downloadTarget(repoWithTag);

		if (this.isDownloadPaused(model)) await this.downloadModel(model);
		else await this.pauseDownload(model);
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
			this.downloadSidecars.delete(event.model);

			return;
		}

		if (request === ModelDownloadStopRequest.PAUSE) {
			// a pause already stored its snapshot when the user asked for it
			if (!this.pausedDownloads.has(event.model)) {
				this.setPausedDownload(event.model, progress);
			}

			this.failedDownloads.delete(event.model);

			return;
		}

		this.deletePausedDownload(event.model);

		const ok = event.event === ServerModelsSseEventType.DOWNLOAD_FINISHED;

		if (ok) {
			this.failedDownloads.delete(event.model);
			this.downloadSidecars.delete(event.model);

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
		return this.pausedDownloads.delete(repoWithTag);
	}

	/** True when the option is the model itself, or a sidecar its download pulls. */
	private pullsSidecar(repoWithTag: string): boolean {
		const { model, sidecar } = downloadTarget(repoWithTag);

		return !sidecar || (this.downloadSidecars.get(model) ?? []).includes(sidecar);
	}

	private rejectStatus(modelId: string, error: Error): void {
		const waiter = this.statusWaiters.get(modelId);

		if (waiter) {
			this.statusWaiters.delete(modelId);
			waiter.reject(error);
		}
	}

	private removeRouterModel(modelId: string): void {
		const idx = this.host.routerModels.findIndex((m) => m.id === modelId);

		if (idx !== -1) {
			this.host.routerModels = this.host.routerModels.filter((m) => m.id !== modelId);
			this.loadProgress.delete(modelId);
			this.downloadProgress.delete(modelId);
			this.failedDownloads.delete(modelId);
			this.deletePausedDownload(modelId);
			this.downloadSidecars.delete(modelId);
			this.stopRequests.delete(modelId);
			this.rejectStatus(modelId, new Error(`Model removed: ${this.host.toDisplayName(modelId)}`));
		}

		// refetch even when the row is unknown: the server removed something our
		// list does not know about, so the list is stale either way
		void this.host.fetchRouterModels();
	}

	private async runStatusReader(signal: AbortSignal): Promise<void> {
		await ModelsService.watchModelEvents(signal, (event) => this.applyStatusEvent(event));
	}

	private setPausedDownload(repoWithTag: string, progress: ModelDownloadProgress | null): void {
		this.pausedDownloads.set(repoWithTag, progress);
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
