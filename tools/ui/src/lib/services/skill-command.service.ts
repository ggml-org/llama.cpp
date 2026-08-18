/** Explicit `/skills <name>` read-then-activate flow. */
import { SkillsService } from '$lib/services/skills.service';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import { skillAvailabilityStore } from '$lib/stores/skill-availability.svelte';
import type { SkillReadResult } from '$lib/types/skills';

/** Outcome of an explicit `/skills <name>` activation. */
export interface SkillCommandOutcome {
	ok: boolean;
	created?: boolean;
	/** True when a NEW durable base activation was persisted. */
	reason?: 'unavailable' | 'not-found' | 'disabled' | 'persistence-failed';
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

	// Check the resolved opaque ID before creating, persisting, or waking.
	if (skillAvailabilityStore.isDisabled(result.skill.id)) {
		return { ok: false, reason: 'disabled' };
	}

	let conversationId = conversationsStore.activeConversation?.id;

	try {
		if (!conversationId) {
			conversationId = await conversationsStore.createConversation(`Skill: ${name}`);
		}

		// Rebuild persisted activations so repeated commands deduplicate.
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
