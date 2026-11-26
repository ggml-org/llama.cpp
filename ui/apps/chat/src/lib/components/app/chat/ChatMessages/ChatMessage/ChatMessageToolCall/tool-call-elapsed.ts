// The live loop timer sits at the end of the last tool row while a reply runs.
// The trace sets it once, naming that row's call, so the per-tool blocks need
// no extra prop.

import { getContext, setContext } from 'svelte';

const KEY = Symbol('tool-call-elapsed');

/** The call that shows the timer and its start (ms since epoch), or null for no timer. */
type ElapsedTarget = { startedAt: number; toolCallId: string } | null;

export function setToolCallElapsedContext(getTarget: () => ElapsedTarget): void {
	setContext(KEY, getTarget);
}

/**
 * Call during component init. Returns a reader of the timer start for a call:
 * its start when that call shows the timer, null otherwise.
 */
export function useToolCallElapsed(): (toolCallId: string | undefined) => number | null {
	const getTarget = getContext<(() => ElapsedTarget) | undefined>(KEY);

	return (toolCallId) => {
		const target = getTarget?.();

		return target && target.toolCallId === toolCallId ? target.startedAt : null;
	};
}
