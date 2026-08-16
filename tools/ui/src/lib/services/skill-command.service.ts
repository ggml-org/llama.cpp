/**
 * Explicit `/skills <name>` activation entry point (slash-command path).
 *
 * Reads first (stateless): a failed read creates no conversation and never
 * wakes. A successful read with no active conversation starts a new
 * `Skill: <name>` conversation carrying the pending CWD. The successful
 * result then routes through the shared durable activation operation - the
 * same path a model-approved read takes. A persistence failure after the
 * read reports an error outcome and never wakes.
 */
import { SkillsService } from '$lib/services/skills.service';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import type { SkillReadResult } from '$lib/types/skills';

/** Outcome of an explicit `/skills <name>` activation. */
export interface SkillCommandOutcome {
	ok: boolean;
	/** True when a NEW durable base activation was persisted. */
	created?: boolean;
	/**
	 * 'unavailable' = the Skills service failed; 'not-found' = the read did
	 * not resolve a base skill; 'persistence-failed' = the read succeeded but
	 * the durable activation could not be persisted.
	 */
	reason?: 'unavailable' | 'not-found' | 'persistence-failed';
}

export async function dispatchSkillActivation(
	name: string,
	options: { cwd?: string; signal?: AbortSignal } = {}
): Promise<SkillCommandOutcome> {
	const cwd =
		options.cwd ??
		conversationsStore.activeConversation?.cwd ??
		conversationsStore.pendingCwd ??
		undefined;

	let result: SkillReadResult;

	try {
		result = await SkillsService.read({ name }, cwd, options.signal);
	} catch {
		return { ok: false, reason: 'unavailable' };
	}

	if (result.kind !== 'skill') {
		return { ok: false, reason: 'not-found' };
	}

	let conversationId = conversationsStore.activeConversation?.id;

	try {
		if (!conversationId) {
			conversationId = await conversationsStore.createConversation(`Skill: ${name}`);
		}

		// Reconstruct the conversation's durable activations from persisted
		// messages before recording, so a repeated `/skills <name>` after a
		// reload reports "already activated" instead of persisting a duplicate
		// synthetic pair.
		await skillActivationStore.loadConversation(conversationId);

		const record = await skillActivationStore.recordActivation({
			conversationId,
			cwd,
			result
		});

		return { created: record.created, ok: true };
	} catch {
		// Rare: persistence failed after a successful read. The conversation
		// (if created) stays visible and deletable; nothing wakes.
		return { ok: false, reason: 'persistence-failed' };
	}
}
