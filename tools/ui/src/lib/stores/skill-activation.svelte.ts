/** Durable Skills activation state shared by model and slash-command paths. */
import { MessageRole, MessageType } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import {
	buildSkillActivationPair,
	isBaseSkillActivation,
	skillActivationExtra,
	skillExtraFromMessage,
	skillResourceExtra
} from '$lib/services/skills-activation.service';
import type {
	SkillActivationInput,
	SkillActivationResult,
	SkillActivationStore
} from '$lib/services/skills-adapters.service';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import type { DatabaseMessage, DatabaseMessageExtraSkill } from '$lib/types/database';
import type { SkillBaseReadResult, SkillReadResult } from '$lib/types/skills';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

/** Activation input narrowed to a base (`kind: 'skill'`) read result. */
type SkillBaseActivationInput = Omit<SkillActivationInput, 'result'> & {
	result: SkillBaseReadResult;
};

export class DurableSkillActivationStore implements SkillActivationStore {
	/** Durable base activations by conversation, keyed by the opaque skill id. */
	private readonly _activatedByConversation = new SvelteMap<string, SvelteSet<string>>();
	private readonly _inFlight = new SvelteMap<string, Promise<DatabaseMessage>>();

	/** Stable per-conversation-per-identity key shared by both activation paths. */
	private static activationKey(conversationId: string, identityId: string): string {
		return `${conversationId}\u0000${identityId}`;
	}

	isActivated(conversationId: string, identityId: string): boolean {
		return this._activatedByConversation.get(conversationId)?.has(identityId) ?? false;
	}

	/** Rebuild the activation cache from valid persisted base records. */
	async loadConversation(conversationId: string): Promise<void> {
		const messages = await conversationsStore.getConversationMessages(conversationId);
		const activated = new SvelteSet<string>();

		for (const message of messages) {
			const extra = skillExtraFromMessage(message);

			if (extra && isBaseSkillActivation(extra)) {
				activated.add(extra.skillId);
			}
		}

		this._activatedByConversation.set(conversationId, activated);
	}

	/** Persist or deduplicate a successful base activation. */
	async recordActivation(input: SkillActivationInput): Promise<SkillActivationResult> {
		const identityId = input.result.skill.id;

		if (this.isActivated(input.conversationId, identityId)) {
			return {
				created: false,
				extra: this.extraFor(input.result),
				toolResultMessage: null
			};
		}

		if (input.result.kind === 'resource') {
			// Resource approval is session-scoped and not persisted.
			this.remember(input.conversationId, identityId);

			return {
				created: false,
				extra: skillResourceExtra(input.result),
				toolResultMessage: null
			};
		}

		// Serialize one activation per conversation and opaque identity.
		const key = DurableSkillActivationStore.activationKey(input.conversationId, identityId);
		const inFlight = this._inFlight.get(key);

		if (inFlight) {
			await inFlight;

			return {
				created: false,
				extra: skillActivationExtra(input.result),
				toolResultMessage: null
			};
		}

		// The resource branch narrows the remaining result to a base read.
		const baseInput: SkillBaseActivationInput = { ...input, result: input.result };
		const transaction =
			input.toolCallId !== undefined
				? this.persistModelActivation(baseInput, input.toolCallId)
				: this.persistSlashActivation(baseInput);

		this._inFlight.set(key, transaction);

		try {
			const toolResultMessage = await transaction;

			this.remember(input.conversationId, identityId);

			return {
				created: true,
				extra: skillActivationExtra(input.result),
				toolResultMessage
			};
		} finally {
			// Clear failures so later activation attempts can retry.
			this._inFlight.delete(key);
		}
	}

	/**
	 * Model path: create the paired tool result under the persisted assistant
	 * message that carries the model's own tool call id.
	 */
	private async persistModelActivation(
		input: SkillBaseActivationInput,
		toolCallId: string
	): Promise<DatabaseMessage> {
		const assistant = await this.resolveAssistantForToolCall(input.conversationId, toolCallId);
		const parentId = assistant?.id ?? (await this.appendParentFor(input.conversationId)) ?? null;
		const resolvedParent =
			parentId ?? (await DatabaseService.createRootMessage(input.conversationId));
		const toolResult = await DatabaseService.createMessageBranch(
			{
				children: [],
				content: input.result.content_xml,
				convId: input.conversationId,
				extra: [skillActivationExtra(input.result)],
				role: MessageRole.TOOL,
				timestamp: Date.now(),
				toolCallId,
				toolCalls: '',
				type: MessageType.TEXT
			},
			resolvedParent
		);

		this.mirrorActive(input.conversationId, toolResult);

		return toolResult;
	}

	/** Slash path: persist the synthetic assistant tool-call + paired tool result atomically. */
	private async persistSlashActivation(input: SkillBaseActivationInput): Promise<DatabaseMessage> {
		const pair = buildSkillActivationPair(input.result, {
			conversationId: input.conversationId,
			cwd: input.cwd
		});
		const parentId = (await this.appendParentFor(input.conversationId)) ?? null;
		const resolvedParent =
			parentId ?? (await DatabaseService.createRootMessage(input.conversationId));
		const [assistant, toolResult] = await DatabaseService.createMessageBranchPair(
			pair.assistant,
			pair.toolResult,
			resolvedParent
		);

		this.mirrorActive(input.conversationId, assistant);
		this.mirrorActive(input.conversationId, toolResult);
		conversationsStore.updateConversationTimestamp();

		return toolResult;
	}

	/** The typed metadata for a non-persisting record (dedupe). */
	private extraFor(result: SkillReadResult): DatabaseMessageExtraSkill {
		return result.kind === 'resource' ? skillResourceExtra(result) : skillActivationExtra(result);
	}

	private remember(conversationId: string, identityId: string): void {
		const set = this._activatedByConversation.get(conversationId) ?? new SvelteSet<string>();

		set.add(identityId);
		this._activatedByConversation.set(conversationId, set);
	}

	/**
	 * Find the persisted assistant message whose tool_calls carry the model's
	 * call id, so the paired tool result keeps a valid transcript pairing.
	 */
	private async resolveAssistantForToolCall(
		conversationId: string,
		toolCallId: string
	): Promise<DatabaseMessage | null> {
		const messages = await conversationsStore.getConversationMessages(conversationId);

		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i];

			if (message.role !== MessageRole.ASSISTANT || !message.toolCalls) continue;

			try {
				const calls = JSON.parse(message.toolCalls) as Array<{ id?: unknown }>;

				if (Array.isArray(calls) && calls.some((call) => call?.id === toolCallId)) {
					return message;
				}
			} catch {
				// Malformed toolCalls on a historical message: skip.
			}
		}

		return null;
	}

	/** Current leaf to append under (last active message, else last persisted, else null). */
	private async appendParentFor(conversationId: string): Promise<string | null> {
		if (conversationsStore.activeConversation?.id === conversationId) {
			const active = conversationsStore.activeMessages;

			if (active.length > 0) return active[active.length - 1].id;
		}

		const messages = await conversationsStore.getConversationMessages(conversationId);
		const last = messages[messages.length - 1];

		if (last) return last.id;

		const root = messages.find((message) => message.parent === null && message.type === 'root');

		return root ? root.id : null;
	}

	/** Mirror a store-created message into the active store when the conversation is displayed. */
	private mirrorActive(conversationId: string, message: DatabaseMessage): void {
		if (conversationsStore.activeConversation?.id === conversationId) {
			conversationsStore.addMessageToActive(message);
		}
	}
}

export const skillActivationStore = new DurableSkillActivationStore();
