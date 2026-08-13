/**
 * Skills activation persistence + presentation helpers.
 *
 * Pure, store-free functions for the shared successful-base-activation
 * operation: the typed durable SKILL metadata record, its validation, the
 * synthetic assistant tool-call + paired tool-result pair used by the
 * explicit `/skills <name>` path, and the reconstruction/rendering helpers
 * that read the durable metadata back. No function here touches host paths,
 * roots, or parses `content_xml`; the XML travels only as opaque tool-result
 * message content.
 */
import { AttachmentType, MessageRole, MessageType } from '$lib/enums';
import { SKILL_READ_TOOL } from '$lib/services/skills-adapters.service';
import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraSkill
} from '$lib/types/database';
import type { SkillBaseReadResult, SkillResourceReadResult } from '$lib/types/skills';
import { uuid } from '$lib/utils';

/** Base message data for the synthetic pair (ids are assigned by the store). */
export interface SkillActivationPairData {
	assistant: Omit<DatabaseMessage, 'id'>;
	toolResult: Omit<DatabaseMessage, 'id'>;
}

/** Safe display facts resolved from a persisted SKILL record for the renderer. */
export interface SkillSectionMeta {
	kind: 'base' | 'resource';
	name: string;
	scope: 'global' | 'project';
	provider: string;
	path?: string;
}

/**
 * Build the typed durable metadata for a successful approved base read.
 * The record holds only server-returned opaque id, safe identity facts, the
 * structured server skill metadata, and the approval/success state.
 */
export function skillActivationExtra(result: SkillBaseReadResult): DatabaseMessageExtraSkill {
	const extra: DatabaseMessageExtraSkill = {
		kind: 'base',
		name: result.skill.name,
		provider: result.skill.provider,
		scope: result.skill.scope,
		skillId: result.skill.id,
		state: 'approved',
		type: AttachmentType.SKILL
	};

	if (result.skill.metadata) {
		extra.metadata = result.skill.metadata;
	}

	return extra;
}

/** Build the typed metadata for an authorized resource result (never an activation). */
export function skillResourceExtra(result: SkillResourceReadResult): DatabaseMessageExtraSkill {
	return {
		kind: 'resource',
		name: result.skill.name,
		path: result.resource.path,
		provider: result.skill.provider,
		scope: result.skill.scope,
		skillId: result.skill.id,
		state: 'approved',
		type: AttachmentType.SKILL
	};
}

/**
 * Type guard for persisted SKILL records. Malformed or unknown historical
 * data fails validation so every consumer (reconstruction, rendering, export
 * fallback) degrades to generic behavior instead of trusting bad shapes.
 */
export function isSkillExtra(value: unknown): value is DatabaseMessageExtraSkill {
	if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;

	const record = value as Partial<DatabaseMessageExtraSkill>;

	if (record.type !== AttachmentType.SKILL) return false;
	if (record.kind !== 'base' && record.kind !== 'resource') return false;
	if (record.state !== 'approved') return false;
	if (typeof record.name !== 'string' || record.name.length === 0) return false;
	if (record.scope !== 'global' && record.scope !== 'project') return false;
	if (typeof record.provider !== 'string' || record.provider.length === 0) return false;
	if (typeof record.skillId !== 'string' || record.skillId.length === 0) return false;

	return true;
}

/** First valid SKILL record in a message's extras, or undefined. */
export function skillExtraFromExtras(
	extras: readonly DatabaseMessageExtra[] | undefined
): DatabaseMessageExtraSkill | undefined {
	if (!extras) return undefined;

	for (const extra of extras) {
		if (isSkillExtra(extra)) return extra;
	}

	return undefined;
}

/** First valid SKILL record persisted on a message, or undefined. */
export function skillExtraFromMessage(message: DatabaseMessage): DatabaseMessageExtraSkill | undefined {
	return skillExtraFromExtras(message.extra);
}

/** True only for persisted base activation records. */
export function isBaseSkillActivation(extra: DatabaseMessageExtraSkill): boolean {
	return extra.kind === 'base';
}

/**
 * Reconstruct an activated identity by its exact opaque id from persisted
 * tool messages. Only valid `kind: 'base'` records count; resource records
 * and malformed extras are ignored.
 */
export function findBaseSkillActivation(
	messages: readonly DatabaseMessage[],
	skillId: string
): DatabaseMessage | undefined {
	for (const message of messages) {
		const extra = skillExtraFromMessage(message);

		if (extra && isBaseSkillActivation(extra) && extra.skillId === skillId) {
			return message;
		}
	}

	return undefined;
}

/**
 * Build the synthetic assistant tool-call message and its paired tool result
 * for an explicit `/skills <name>` activation. Both messages share one
 * generated tool call id so the pair stays a valid model transcript after
 * reload; the tool result carries the typed base-activation metadata.
 */
export function buildSkillActivationPair(
	result: SkillBaseReadResult,
	options: { conversationId: string; cwd?: string; toolCallId?: string }
): SkillActivationPairData {
	const toolCallId = options.toolCallId ?? `skill_${uuid()}`;

	return {
		assistant: {
			children: [],
			content: '',
			convId: options.conversationId,
			parent: null,
			role: MessageRole.ASSISTANT,
			timestamp: Date.now(),
			toolCalls: JSON.stringify([
				{
					id: toolCallId,
					type: 'function',
					function: {
						name: SKILL_READ_TOOL,
						arguments: JSON.stringify({ name: result.skill.name })
					}
				}
			]),
			type: MessageType.TEXT
		},
		toolResult: {
			children: [],
			content: result.content_xml,
			convId: options.conversationId,
			extra: [skillActivationExtra(result)],
			parent: null,
			role: MessageRole.TOOL,
			timestamp: Date.now() + 1,
			toolCallId,
			toolCalls: '',
			type: MessageType.TEXT
		}
	};
}

/**
 * Resolve safe display metadata for a tool-call section. Returns undefined —
 * so the generic tool card renders — for non-Skills tools, missing extras,
 * or malformed historical records. Labels come only from resolved metadata.
 */
export function resolveSkillSectionMeta(section: {
	toolName?: string;
	toolResultExtras?: DatabaseMessageExtra[];
}): SkillSectionMeta | undefined {
	if (section.toolName !== SKILL_READ_TOOL) return undefined;

	const extra = skillExtraFromExtras(section.toolResultExtras);

	if (!extra) return undefined;

	const meta: SkillSectionMeta = {
		kind: extra.kind,
		name: extra.name,
		provider: extra.provider,
		scope: extra.scope
	};

	if (extra.path !== undefined) meta.path = extra.path;

	return meta;
}

/** True when the section is a Skills result with valid persisted metadata. */
export function isSkillToolSection(section: {
	toolName?: string;
	toolResultExtras?: DatabaseMessageExtra[];
}): boolean {
	return resolveSkillSectionMeta(section) !== undefined;
}
