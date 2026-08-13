/**
 * Explicit `/skills <name>` activation entry point (slash-command path).
 *
 * Resolves the base read through the server, then routes the successful
 * result through the shared durable activation operation — the same path a
 * model-approved read takes. Unavailable/not-found/error reads persist
 * nothing.
 */
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import { SkillsService } from '$lib/services/skills.service';
import type { SkillReadResult } from '$lib/types/skills';

/** Outcome of an explicit `/skills <name>` activation. */
export interface SkillCommandOutcome {
	ok: boolean;
	/** True when a NEW durable base activation was persisted. */
	created?: boolean;
	/** 'unavailable' = the Skills service failed; 'not-found' = the read did not resolve a base skill. */
	reason?: 'unavailable' | 'not-found';
}

export async function activateSkillByName(
	conversationId: string,
	name: string,
	options: { cwd?: string; signal?: AbortSignal } = {}
): Promise<SkillCommandOutcome> {
	let result: SkillReadResult;

	try {
		result = await SkillsService.read({ name }, options.cwd, options.signal);
	} catch {
		return { ok: false, reason: 'unavailable' };
	}

	if (result.kind !== 'skill') {
		return { ok: false, reason: 'not-found' };
	}

	const record = await skillActivationStore.recordActivation({
		conversationId,
		cwd: options.cwd,
		result
	});

	return { created: record.created, ok: true };
}
