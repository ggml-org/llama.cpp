/** Route-owned, abortable Skills catalog refresh lifecycle. */
import { skillsStore } from '$lib/stores/skills.svelte';

export interface SkillCatalogRefresh {
	/** Handle the initial mount or a selected-CWD change. */
	onCwdChange(cwd: string | undefined): void;
	/** Force a fresh request for the current CWD. */
	retry(): void;
	/** Abort the in-flight request. */
	dispose(): void;
}

export function useSkillCatalogRefresh(): SkillCatalogRefresh {
	let currentCwd: string | undefined;
	let inflight: AbortController | undefined;
	let initialized = false;
	let disposed = false;

	function request(cwd: string | undefined, force: boolean): void {
		if (disposed) return;

		if (initialized && cwd !== currentCwd) {
			// Invalidate the previous route slot before its response can apply.
			skillsStore.invalidate(currentCwd);
		}

		initialized = true;
		currentCwd = cwd;

		inflight?.abort();

		const next = new AbortController();

		inflight = next;

		// The slot records success or failure; this promise need not be awaited.
		const load = force
			? skillsStore.refresh(cwd, next.signal)
			: skillsStore.ensureCatalog(cwd, next.signal);

		void load.catch(() => {});
	}

	return {
		dispose() {
			if (disposed) return;

			disposed = true;
			inflight?.abort();
		},
		onCwdChange(cwd) {
			if (disposed) return;

			if (initialized && cwd === currentCwd) return;

			request(cwd, false);
		},
		retry() {
			if (disposed || !initialized) return;

			request(currentCwd, true);
		}
	};
}
