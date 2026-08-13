/**
 * skillsStore - CWD-keyed latest catalog screen state and per-run snapshots.
 *
 * Each selected-CWD string owns an independent screen slot and a monotonic
 * request generation. A stale response (an older generation resolving late) is
 * still returned to its issuing caller but can never replace the current slot.
 * Run snapshots are built from the run's own response, never the mutable slot,
 * so concurrent runs and CWD changes cannot swap a run's CWD/catalog data.
 */
import { SvelteMap } from 'svelte/reactivity';
import { buildSkillRunSnapshot } from '$lib/services/skills-packing.service';
import { SkillsService } from '$lib/services/skills.service';
import type { SkillCatalogResponse, SkillRunSnapshot } from '$lib/types';

export type SkillCatalogStatus = 'loading' | 'ready' | 'error';

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

class SkillsStore {
	private _slots = $state<SvelteMap<string | undefined, SkillCatalogSlot>>(new SvelteMap());
	private _generationByCwd = new Map<string | undefined, number>();

	/** Current screen slot for a selected CWD (undefined = no CWD selected). */
	slotFor(cwd: string | undefined): SkillCatalogSlot | undefined {
		return this._slots.get(cwd);
	}

	/**
	 * Fetch the catalog for the screen. Bumps the CWD's generation; the result
	 * is always returned to this caller, but only replaces the UI slot while it
	 * remains the latest generation for that CWD.
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
	 * Create an immutable per-run snapshot from the run's OWN catalog response.
	 * Never reads or writes the mutable screen slot, so concurrent runs and
	 * CWD changes cannot swap a run's CWD/catalog data.
	 */
	async createRunSnapshot(cwd: string | undefined, signal?: AbortSignal): Promise<SkillRunSnapshot> {
		const catalog = await SkillsService.list(cwd, signal);

		return buildSkillRunSnapshot(cwd, catalog);
	}

	/**
	 * Immediately invalidate the screen state for a CWD (e.g. the selected CWD
	 * changed). Bumps the generation so any in-flight response for the key
	 * becomes stale; frozen run snapshots are unaffected.
	 */
	invalidate(cwd: string | undefined): void {
		this.bumpGeneration(cwd);
		this._slots.delete(cwd);
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

export const skillsStore = new SkillsStore();
