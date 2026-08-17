/** Persistence and presentation helpers for Skills activations. */
import { SKILL_READ_TOOL } from '$lib/constants';
import { AttachmentType, MessageRole, MessageType } from '$lib/enums';
import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraSkill
} from '$lib/types/database';
import type { SkillBaseReadResult, SkillResourceReadResult } from '$lib/types/skills';
import { uuid } from '$lib/utils';

/** Data for the synthetic assistant/tool-result pair. */
export interface SkillActivationPairData {
	assistant: Omit<DatabaseMessage, 'id'>;
	toolResult: Omit<DatabaseMessage, 'id'>;
}

/** Safe display facts from persisted Skills metadata. */
export interface SkillSectionMeta {
	kind: 'base' | 'resource';
	name: string;
	scope: 'global' | 'project';
	provider: string;
	path?: string;
}

/** Build durable metadata for a successful base read. */
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

/** Validate persisted SKILL metadata before exposing it to consumers. */
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
export function skillExtraFromMessage(
	message: DatabaseMessage
): DatabaseMessageExtraSkill | undefined {
	return skillExtraFromExtras(message.extra);
}

/** True only for persisted base activation records. */
export function isBaseSkillActivation(extra: DatabaseMessageExtraSkill): boolean {
	return extra.kind === 'base';
}

/** Find a valid base activation for an opaque identity. */
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

/** Build the synthetic assistant/tool-result pair for explicit activation. */
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
					function: {
						arguments: JSON.stringify({ name: result.skill.name }),
						name: SKILL_READ_TOOL
					},
					id: toolCallId,
					type: 'function'
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
