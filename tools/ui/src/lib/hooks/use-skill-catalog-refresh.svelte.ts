/**
 * useSkillCatalogRefresh - route-owned abortable catalog refresh lifecycle.
 *
 * Couples selected-CWD changes to immediate route-slot invalidation of the
 * previous CWD plus an abortable refresh for the new one. The route component
 * calls `onCwdChange` with the current CWD on every reactive change; repeated
 * identical values are no-ops, so the route's `$effect` may re-run without
 * duplicating requests. `retry` forces a fresh request for the current CWD
 * (error-state refresh) and `dispose` aborts any in-flight request when the
 * route unmounts.
 *
 * Frozen agent run snapshots are never touched: invalidation only bumps the
 * per-CWD generation and drops the mutable screen slot (see skillsStore).
 */
import { skillsStore } from '$lib/stores/skills.svelte';

export interface SkillCatalogRefresh {
	/** React to a selected-CWD change (including the initial mount). */
	onCwdChange(cwd: string | undefined): void;
	/** Force a fresh request for the current CWD, aborting any in-flight one. */
	retry(): void;
	/** Abort the in-flight request; the controller becomes inert. */
	dispose(): void;
}

export function useSkillCatalogRefresh(): SkillCatalogRefresh {
	let currentCwd: string | undefined;
	let inflight: AbortController | undefined;
	let initialized = false;
	let disposed = false;

	function request(cwd: string | undefined): void {
		if (disposed) return;

		if (initialized && cwd !== currentCwd) {
			// A selected-CWD change invalidates the previous route slot
			// immediately; any in-flight response for it becomes stale.
			skillsStore.invalidate(currentCwd);
		}

		initialized = true;
		currentCwd = cwd;

		inflight?.abort();

		const next = new AbortController();

		inflight = next;

		// The slot records the outcome and the route renders from it, so the
		// promise is intentionally unawaited; rejections (abort or error) are
		// already reflected in the slot.
		void skillsStore.refresh(cwd, next.signal).catch(() => {});
	}

	return {
		onCwdChange(cwd) {
			if (disposed) return;

			if (initialized && cwd === currentCwd) return;

			request(cwd);
		},
		retry() {
			if (disposed || !initialized) return;

			request(currentCwd);
		},
		dispose() {
			if (disposed) return;

			disposed = true;
			inflight?.abort();
		}
	};
}
