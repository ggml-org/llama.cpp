import { SKILL_READ_TOOL } from '$lib/constants';
import { AttachmentType, MessageRole, MessageType } from '$lib/enums';
import {
	buildSkillActivationPair,
	findBaseSkillActivation,
	isBaseSkillActivation,
	isSkillExtra,
	isSkillToolSection,
	resolveSkillSectionMeta,
	skillActivationExtra,
	skillExtraFromExtras,
	skillExtraFromMessage,
	skillResourceExtra
} from '$lib/services/skills-activation.service';
import type { DatabaseMessage, DatabaseMessageExtra, DatabaseMessageExtraSkill } from '$lib/types';
import type {
	SkillBaseReadResult,
	SkillMetadata,
	SkillResourceReadResult
} from '$lib/types/skills';
import { describe, expect, it } from 'vitest';

const METADATA: SkillMetadata = {
	name: 'demo-skill',
	description: 'A demo skill',
	license: 'MIT'
};

function baseResult(overrides: Partial<SkillBaseReadResult> = {}): SkillBaseReadResult {
	return {
		kind: 'skill',
		skill: {
			id: 'opaque-id-1',
			name: 'demo-skill',
			scope: 'project',
			provider: 'agents',
			metadata: METADATA
		},
		resources: { paths: ['refs/DETAILS.md'], truncated: false },
		source: '---\nname: demo-skill\ndescription: A demo skill\n---\n# Body',
		body_markdown: '# Body',
		content_xml: '<skill_content name="demo-skill">body &amp; more</skill_content>',
		diagnostics: [],
		...overrides
	};
}

function resourceResult(overrides: Partial<SkillResourceReadResult> = {}): SkillResourceReadResult {
	return {
		kind: 'resource',
		skill: { id: 'opaque-id-1', name: 'demo-skill', scope: 'project', provider: 'agents' },
		resource: { path: 'refs/DETAILS.md' },
		content_xml: '<skill_resource>data</skill_resource>',
		diagnostics: [],
		...overrides
	};
}

function toolMessage(extra?: DatabaseMessageExtraSkill, id = 'msg-1'): DatabaseMessage {
	return {
		children: [],
		content: '<skill_content>x</skill_content>',
		convId: 'conv-1',
		id,
		parent: 'assistant-1',
		role: MessageRole.TOOL,
		timestamp: 1,
		toolCallId: 'call_1',
		toolCalls: '',
		type: MessageType.TEXT,
		...(extra ? { extra: [extra] } : {})
	} as DatabaseMessage;
}

describe('skillActivationExtra', () => {
	it('builds a typed durable base record with only safe server-returned fields', () => {
		const extra = skillActivationExtra(baseResult());

		expect(extra.type).toBe(AttachmentType.SKILL);
		expect(extra.kind).toBe('base');
		expect(extra.state).toBe('approved');
		expect(extra.skillId).toBe('opaque-id-1');
		expect(extra.name).toBe('demo-skill');
		expect(extra.scope).toBe('project');
		expect(extra.provider).toBe('agents');
		expect(extra.metadata).toEqual(METADATA);
		expect(extra.path).toBeUndefined();
	});

	it('never stores content_xml, resource paths, host paths, or roots in the record', () => {
		const extra = skillActivationExtra(baseResult());

		expect(JSON.stringify(extra)).not.toContain('content_xml');
		expect(JSON.stringify(extra)).not.toContain('refs/DETAILS.md');
		expect(JSON.stringify(extra)).not.toContain('/home/');
		expect(JSON.stringify(extra)).not.toContain('cwd');
		expect('content' in extra).toBe(false);
	});

	it('omits the metadata field when the server returned none', () => {
		const extra = skillActivationExtra(baseResult({ skill: { id: 'x', name: 'n', scope: 'global', provider: 'p' } }));

		expect(extra.metadata).toBeUndefined();
	});
});

describe('skillResourceExtra', () => {
	it('builds a resource record carrying the requested relative path', () => {
		const extra = skillResourceExtra(resourceResult());

		expect(extra.type).toBe(AttachmentType.SKILL);
		expect(extra.kind).toBe('resource');
		expect(extra.state).toBe('approved');
		expect(extra.skillId).toBe('opaque-id-1');
		expect(extra.path).toBe('refs/DETAILS.md');
		expect(extra.metadata).toBeUndefined();
		expect(JSON.stringify(extra)).not.toContain('content_xml');
	});
});

describe('isSkillExtra', () => {
	it('accepts valid base and resource records', () => {
		expect(isSkillExtra(skillActivationExtra(baseResult()))).toBe(true);
		expect(isSkillExtra(skillResourceExtra(resourceResult()))).toBe(true);
	});

	it('rejects null, non-objects, and non-SKILL extras', () => {
		expect(isSkillExtra(null)).toBe(false);
		expect(isSkillExtra(undefined)).toBe(false);
		expect(isSkillExtra('nope')).toBe(false);
		expect(isSkillExtra(42)).toBe(false);
		expect(isSkillExtra({ type: AttachmentType.TEXT, name: 'x' })).toBe(false);
	});

	it('rejects malformed SKILL-shaped records (missing or invalid required fields)', () => {
		const valid = skillActivationExtra(baseResult());
		const { skillId: _skillId, ...missingId } = valid;

		expect(isSkillExtra({ ...valid, skillId: undefined })).toBe(false);
		expect(isSkillExtra({ ...valid, skillId: 7 })).toBe(false);
		expect(isSkillExtra({ ...valid, kind: 'other' })).toBe(false);
		expect(isSkillExtra({ ...valid, state: 'denied' })).toBe(false);
		expect(isSkillExtra({ ...valid, name: '' })).toBe(false);
		expect(isSkillExtra({ ...valid, scope: 'host' })).toBe(false);
		expect(isSkillExtra(missingId)).toBe(false);
	});
});

describe('skillExtraFromExtras / skillExtraFromMessage', () => {
	it('returns the first valid SKILL extra, ignoring unrelated and malformed entries', () => {
		const valid = skillActivationExtra(baseResult());

		expect(
			skillExtraFromExtras([
				{ type: AttachmentType.TEXT, name: 't', content: 'c' },
				{ ...valid, skillId: undefined } as unknown as DatabaseMessageExtra,
				valid
			])
		).toEqual(valid);
	});

	it('returns undefined for empty, undefined, or all-invalid extras', () => {
		expect(skillExtraFromExtras(undefined)).toBeUndefined();
		expect(skillExtraFromExtras([])).toBeUndefined();
		expect(
			skillExtraFromExtras([{ type: AttachmentType.IMAGE, name: 'i', base64Url: 'data:image/png;base64,AA==' }])
		).toBeUndefined();
	});

	it('reads the SKILL extra from a persisted message', () => {
		const valid = skillActivationExtra(baseResult());

		expect(skillExtraFromMessage(toolMessage(valid))).toEqual(valid);
		expect(skillExtraFromMessage(toolMessage())).toBeUndefined();
	});
});

describe('isBaseSkillActivation', () => {
	it('distinguishes durable base records from resource records', () => {
		expect(isBaseSkillActivation(skillActivationExtra(baseResult()))).toBe(true);
		expect(isBaseSkillActivation(skillResourceExtra(resourceResult()))).toBe(false);
	});
});

describe('findBaseSkillActivation', () => {
	it('reconstructs an activated identity by its exact opaque id from persisted tool messages', () => {
		const extra = skillActivationExtra(baseResult());
		const messages = [
			toolMessage(),
			toolMessage(extra, 'msg-2'),
			toolMessage(skillResourceExtra(resourceResult()), 'msg-3')
		];

		expect(findBaseSkillActivation(messages, 'opaque-id-1')?.id).toBe('msg-2');
	});

	it('ignores resource records and malformed extras for reconstruction', () => {
		const messages = [
			toolMessage(skillResourceExtra(resourceResult())),
			toolMessage({ ...skillActivationExtra(baseResult()), skillId: undefined } as unknown as DatabaseMessageExtraSkill)
		];

		expect(findBaseSkillActivation(messages, 'opaque-id-1')).toBeUndefined();
	});

	it('returns undefined when the opaque id was never activated', () => {
		expect(findBaseSkillActivation([toolMessage(skillActivationExtra(baseResult()))], 'other-id')).toBeUndefined();
		expect(findBaseSkillActivation([], 'opaque-id-1')).toBeUndefined();
	});
});

describe('buildSkillActivationPair', () => {
	it('builds a valid synthetic assistant tool-call message paired with its tool result', () => {
		const pair = buildSkillActivationPair(baseResult(), { conversationId: 'conv-1', cwd: undefined });

		expect(pair.assistant.role).toBe(MessageRole.ASSISTANT);
		expect(pair.assistant.convId).toBe('conv-1');
		expect(pair.assistant.content).toBe('');
		expect(pair.assistant.type).toBe(MessageType.TEXT);
		expect(pair.toolResult.role).toBe(MessageRole.TOOL);
		expect(pair.toolResult.convId).toBe('conv-1');
		expect(pair.toolResult.content).toBe('<skill_content name="demo-skill">body &amp; more</skill_content>');

		const calls = JSON.parse(pair.assistant.toolCalls ?? '') as Array<{
			id: string;
			type: string;
			function: { name: string; arguments: string };
		}>;

		expect(calls).toHaveLength(1);
		expect(calls[0].function.name).toBe(SKILL_READ_TOOL);
		expect(JSON.parse(calls[0].function.arguments)).toEqual({ name: 'demo-skill' });
		expect(pair.toolResult.toolCallId).toBe(calls[0].id);
	});

	it('records the typed base activation metadata on the paired tool result', () => {
		const pair = buildSkillActivationPair(baseResult(), { conversationId: 'conv-1' });
		const [extra] = pair.toolResult.extra ?? [];

		expect(isSkillExtra(extra)).toBe(true);
		if (isSkillExtra(extra)) {
			expect(extra.kind).toBe('base');
			expect(extra.skillId).toBe('opaque-id-1');
		}
		expect(JSON.stringify(pair)).not.toContain('content_xml');
	});

	it('never carries host paths or roots in either message', () => {
		const pair = buildSkillActivationPair(baseResult(), { conversationId: 'conv-1', cwd: '/home/user' });

		expect(pair.toolResult.toolCwd).toBeUndefined();
		expect(JSON.stringify(pair)).not.toContain('/home/user');
		expect(JSON.stringify(pair)).not.toContain('refs/DETAILS.md');
	});
});

describe('resolveSkillSectionMeta / isSkillToolSection', () => {
	it('resolves safe display metadata from a valid base record', () => {
		const section = {
			toolName: SKILL_READ_TOOL,
			toolResultExtras: [skillActivationExtra(baseResult())]
		};

		expect(resolveSkillSectionMeta(section)).toEqual({
			kind: 'base',
			name: 'demo-skill',
			scope: 'project',
			provider: 'agents'
		});
		expect(isSkillToolSection(section)).toBe(true);
	});

	it('resolves the requested resource path from a resource record', () => {
		const section = {
			toolName: SKILL_READ_TOOL,
			toolResultExtras: [skillResourceExtra(resourceResult())]
		};

		expect(resolveSkillSectionMeta(section)).toEqual({
			kind: 'resource',
			name: 'demo-skill',
			scope: 'project',
			provider: 'agents',
			path: 'refs/DETAILS.md'
		});
	});

	it('falls back to generic rendering for unknown tools, missing, or malformed metadata', () => {
		expect(resolveSkillSectionMeta({ toolName: 'other_tool', toolResultExtras: [skillActivationExtra(baseResult())] })).toBeUndefined();
		expect(resolveSkillSectionMeta({ toolName: SKILL_READ_TOOL, toolResultExtras: [] })).toBeUndefined();
		expect(resolveSkillSectionMeta({ toolName: SKILL_READ_TOOL, toolResultExtras: [{ type: AttachmentType.TEXT, name: 'x', content: 'x' }] })).toBeUndefined();
		expect(resolveSkillSectionMeta({ toolName: SKILL_READ_TOOL })).toBeUndefined();
		expect(isSkillToolSection({ toolName: SKILL_READ_TOOL })).toBe(false);
	});
});
