/**
 * ChatActivityStore - Conversation activity ledger
 *
 * Single owner of the "is this conversation doing something" state:
 * - `local`  - this browser is piping a stream (send, server-stream attach,
 *              or resume-wait while the owning model loads)
 * - `remote` - the backend reports a running session, no local pipe yet
 *              (global snapshot on mount / visibilitychange)
 *
 * The union of both drives the sidebar spinners (`loadingConvs`); `local`
 * drives the per-conversation loading flags. When a local pipe ends it is
 * the authoritative observer of session end, so it also drops the stale
 * remote hint in the same call - no cross-owner cleanup, no ghosted
 * spinners waiting for the next visibilitychange snapshot.
 *
 * Composed under chatStore.activity; not exported from the stores barrel.
 */

import { SvelteMap, SvelteSet } from 'svelte/reactivity';

export class ChatActivityStore {
	/** Convs this browser is piping a stream for (send, attach, resume-wait). */
	private local = new SvelteSet<string>();
	/** Convs the backend reports as having a running session (snapshot sync). */
	private remote = new SvelteSet<string>();
	/** Convs with any activity, the union the sidebar spinners render. */
	loadingConvs = $derived.by(() => {
		const out = new SvelteSet<string>(this.local);

		for (const id of this.remote) out.add(id);

		return Array.from(out);
	});

	/**
	 * Convs whose in-flight run is a compaction summary, mapped to the node
	 * being summarized. That row reports the progress itself, so the chat
	 * turns it summarizes must not repeat the same indicator.
	 */
	private compacting = new SvelteMap<string, string>();

	/**
	 * Apply a backend snapshot of running sessions (mount / visibilitychange).
	 * Diffed so unchanged entries do not re-trigger reactivity.
	 */
	applyRemoteSnapshot(running: Iterable<string>): void {
		const next = new SvelteSet<string>(running);

		for (const id of Array.from(this.remote)) {
			if (!next.has(id)) this.remote.delete(id);
		}

		for (const id of next) this.remote.add(id);
	}

	/** The conv's compaction summary ended, however it ended. */
	endCompacting(convId: string): void {
		this.compacting.delete(convId);
	}

	/** Node id of the summary running for the conv, or null when none runs. */
	getCompactingNodeId(convId: string): string | null {
		return this.compacting.get(convId) ?? null;
	}

	isCompacting(convId: string): boolean {
		return this.compacting.has(convId);
	}

	isLocal(convId: string): boolean {
		return this.local.has(convId);
	}

	isRemote(convId: string): boolean {
		return this.remote.has(convId);
	}

	/**
	 * A local pipe ended for the conv. Also drops the remote hint: the local
	 * pipe is the authoritative observer of session end, so the sidebar hint
	 * goes away right away instead of ghosting until the next snapshot.
	 */
	localEnded(convId: string): void {
		this.local.delete(convId);
		this.remote.delete(convId);
	}

	/** A compaction summary started for the conv. */
	markCompacting(convId: string, nodeId: string): void {
		this.compacting.set(convId, nodeId);
	}

	/** A local pipe (send, attach or resume-wait) started for the conv. */
	markLocal(convId: string): void {
		this.local.add(convId);
	}
}

export const chatActivityStore = new ChatActivityStore();
