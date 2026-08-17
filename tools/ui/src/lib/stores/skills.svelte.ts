/** CWD-keyed Skills catalog state and startup navigation availability. */
import { SkillsService, buildSkillRunSnapshot } from '$lib/services/skills.service';
import { skillAvailabilityStore } from '$lib/stores/skill-availability.svelte';
import type { SkillCatalogResponse, SkillRunSnapshot } from '$lib/types';
import { ApiError } from '$lib/utils/api-fetch';
import { SvelteMap } from 'svelte/reactivity';

export type SkillCatalogStatus = 'loading' | 'ready' | 'error';

/** Startup gate for showing the Skills entry in the sidebar navigation. */
export type SkillAvailability = 'unknown' | 'loading' | 'available' | 'disabled' | 'error';

/** Latest catalog screen state for one selected-CWD key. */
export interface SkillCatalogSlot {
	status: SkillCatalogStatus;
	/** Monotonic generation of the request that produced this slot (per CWD). */
	generation: number;
	/** Selected CWD this slot was resolved under; undefined = server process CWD. */
	cwd: string | undefined;
	/** Latest successful catalog response; present when status === 'ready'. */
	catalog?: SkillCatalogResponse;
	/** Failure from the latest request; present when status === 'error'. */
	error?: unknown;
}

/** One shared (or torn-down) catalog request for a single CWD key. */
interface CatalogEnsure {
	/** Store-owned abort; aborted only after the final subscriber leaves. */
	controller: AbortController;
	/** The single shared catalog request for this CWD. */
	request: Promise<SkillCatalogResponse>;
	/** Signals of callers still waiting; empty means none are attached. */
	subscribers: Set<AbortSignal | undefined>;
	/** True once the request settled or the last subscriber tore the entry down. */
	settled: boolean;
	/** CWD this entry was created for. */
	cwd: string | undefined;
	/** Generation captured at creation; guards slot writes against newer refreshes. */
	generation: number;
}

class SkillsStore {
	private _slots = $state<SvelteMap<string | undefined, SkillCatalogSlot>>(new SvelteMap());
	private _generationByCwd = new Map<string | undefined, number>();
	private _availability = $state<SkillAvailability>('unknown');
	private _probeGeneration = 0;
	private _catalogEnsures = new Map<string | undefined, CatalogEnsure>();

	/** Startup Skills navigation availability; see `showInNavigation`. */
	get availability(): SkillAvailability {
		return this._availability;
	}

	/** The Skills entry is shown once the catalog is confirmed or a non-404 failure needs retry. */
	get showInNavigation(): boolean {
		return this._availability === 'available' || this._availability === 'error';
	}

	/** Current screen slot for a selected CWD (undefined = no CWD selected). */
	slotFor(cwd: string | undefined): SkillCatalogSlot | undefined {
		return this._slots.get(cwd);
	}

	/**
	 * Fetch the catalog. The result always returns to the caller but updates
	 * the UI slot only while it is still the latest generation for that CWD.
	 */
	async refresh(cwd: string | undefined, signal?: AbortSignal): Promise<SkillCatalogResponse> {
		const generation = this.bumpGeneration(cwd);

		this.setSlot({ cwd, generation, status: 'loading' });

		try {
			const catalog = await SkillsService.list(cwd, signal);

			if (this._generationByCwd.get(cwd) === generation) {
				this.setSlot({ catalog, cwd, generation, status: 'ready' });
			}

			return catalog;
		} catch (error) {
			if (this._generationByCwd.get(cwd) === generation) {
				this.setSlot({ cwd, error, generation, status: 'error' });
			}

			throw error;
		}
	}

	/**
	 * Startup and initial route loading: one shared request per CWD. Aborting
	 * one caller rejects only that caller; the shared request is aborted only
	 * after its final subscriber leaves.
	 */
	async ensureCatalog(
		cwd: string | undefined,
		signal?: AbortSignal
	): Promise<SkillCatalogResponse> {
		const existing = this._catalogEnsures.get(cwd);

		if (existing && !existing.settled) {
			return this.attachToEnsure(cwd, existing, signal);
		}

		const generation = this.bumpGeneration(cwd);
		const controller = new AbortController();
		const request = SkillsService.list(cwd, controller.signal);
		const entry: CatalogEnsure = {
			controller,
			cwd,
			generation,
			request,
			settled: false,
			subscribers: new Set()
		};

		this._catalogEnsures.set(cwd, entry);
		this.setSlot({ cwd, generation, status: 'loading' });

		void request.then(
			(catalog) => {
				entry.settled = true;

				if (this._catalogEnsures.get(cwd) === entry) {
					this._catalogEnsures.delete(cwd);
				}

				if (this._generationByCwd.get(cwd) === generation) {
					this.setSlot({ catalog, cwd, generation, status: 'ready' });
				}
			},
			(error) => {
				entry.settled = true;

				if (this._catalogEnsures.get(cwd) === entry) {
					this._catalogEnsures.delete(cwd);
				}

				if (this._generationByCwd.get(cwd) === generation) {
					this.setSlot({ cwd, error, generation, status: 'error' });
				}
			}
		);

		return this.attachToEnsure(cwd, entry, signal);
	}

	/**
	 * Probe the catalog and gate the sidebar entry: success -> available,
	 * only 404 -> disabled, every other failure -> error.
	 */
	async probeAvailability(cwd: string | undefined, signal?: AbortSignal): Promise<void> {
		const generation = ++this._probeGeneration;

		this._availability = 'loading';

		try {
			await this.ensureCatalog(cwd, signal);

			if (this._probeGeneration === generation) {
				this._availability = 'available';
			}
		} catch (error) {
			if (this._probeGeneration !== generation) return;

			// An aborted caller changes nothing; the abort still propagates.
			if (signal?.aborted || isAbortError(error)) {
				throw error;
			}

			if (error instanceof ApiError && error.status === 404) {
				this._availability = 'disabled';
			} else {
				this._availability = 'error';
			}
		}
	}

	/** Create an immutable per-run snapshot from the run's own catalog response. */
	async createRunSnapshot(
		cwd: string | undefined,
		signal?: AbortSignal
	): Promise<SkillRunSnapshot> {
		const catalog = await SkillsService.list(cwd, signal);

		return buildSkillRunSnapshot(cwd, catalog, skillAvailabilityStore.disabledIds);
	}

	/** Invalidate screen state for a CWD; in-flight responses become stale. */
	invalidate(cwd: string | undefined): void {
		this.bumpGeneration(cwd);
		this._slots.delete(cwd);
	}

	/** Attach one caller to a shared entry; aborting it rejects only this caller. */
	private attachToEnsure(
		cwd: string | undefined,
		entry: CatalogEnsure,
		signal?: AbortSignal
	): Promise<SkillCatalogResponse> {
		if (signal?.aborted) {
			return Promise.reject(abortError());
		}

		const { promise, reject, resolve } = Promise.withResolvers<SkillCatalogResponse>();
		const onAbort = () => {
			entry.subscribers.delete(signal);

			if (entry.subscribers.size === 0 && !entry.settled) {
				entry.settled = true;

				if (this._catalogEnsures.get(cwd) === entry) {
					this._catalogEnsures.delete(cwd);
				}

				entry.controller.abort();
			}

			reject(abortError());
		};

		entry.subscribers.add(signal);

		if (signal) {
			signal.addEventListener('abort', onAbort, { once: true });
		}

		void entry.request.then(
			(catalog) => {
				signal?.removeEventListener('abort', onAbort);
				resolve(catalog);
			},
			(error) => {
				signal?.removeEventListener('abort', onAbort);
				reject(error);
			}
		);

		return promise;
	}

	/** Advance the per-CWD monotonic request generation and return it. */
	private bumpGeneration(cwd: string | undefined): number {
		const generation = (this._generationByCwd.get(cwd) ?? 0) + 1;

		this._generationByCwd.set(cwd, generation);

		return generation;
	}

	private setSlot(slot: SkillCatalogSlot): void {
		this._slots.set(slot.cwd, slot);
	}
}

/** Standard abort rejection shared by all store-owned request teardown. */
function abortError(): DOMException {
	return new DOMException('The operation was aborted', 'AbortError');
}

function isAbortError(error: unknown): boolean {
	return error instanceof DOMException && error.name === 'AbortError';
}

export const skillsStore = new SkillsStore();
