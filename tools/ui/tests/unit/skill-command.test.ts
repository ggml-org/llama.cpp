import { getChatCommands } from '$lib/utils';
import { AttachmentType, ChatFormCommandAction } from '$lib/enums';
import { activateSkillByName } from '$lib/services/skill-command.service';
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import { SkillsService } from '$lib/services/skills.service';
import type { SkillBaseReadResult } from '$lib/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/skills.service', () => ({
	SkillsService: { list: vi.fn(), read: vi.fn() }
}));
vi.mock('$lib/stores/skill-activation.svelte', () => ({
	skillActivationStore: {
		isActivated: vi.fn(() => false),
		loadConversation: vi.fn().mockResolvedValue(undefined),
		recordActivation: vi.fn()
	}
}));

const mockRead = vi.mocked(SkillsService.read);
const mockRecordActivation = vi.mocked(skillActivationStore.recordActivation);
const mockLoadConversation = vi.mocked(skillActivationStore.loadConversation);

function baseResult(name: string): SkillBaseReadResult {
	return {
		kind: 'skill',
		skill: { id: `opaque-${name}`, name, scope: 'project', provider: 'agents' },
		resources: { paths: [], truncated: false },
		source: `---\nname: ${name}\n---\n# Body`,
		body_markdown: '# Body',
		content_xml: `<skill_content name="${name}">body</skill_content>`,
		diagnostics: []
	};
}

beforeEach(() => {
	vi.clearAllMocks();
	mockRecordActivation.mockResolvedValue({
		created: true,
		extra: { type: AttachmentType.SKILL, kind: 'base', state: 'approved', name: 'demo-skill', scope: 'project', provider: 'agents', skillId: 'opaque-demo-skill' },
		toolResultMessage: null
	});
});

describe('getChatCommands /skills', () => {
	it('surfaces the /skills command through the existing command discovery path', () => {
		const commands = getChatCommands({
			hasCwdTools: () => true,
			hasPrompts: () => true,
			hasSkills: () => true,
			showModelSelector: true
		});
		const skills = commands.find((c) => c.action === ChatFormCommandAction.SKILLS);

		expect(skills).toBeDefined();
		expect(skills?.name).toBe('skills');
		expect(skills?.disabled).toBe(false);
		expect(skills?.description.length).toBeGreaterThan(0);
	});

	it('disables /skills when the capability is unavailable', () => {
		const commands = getChatCommands({
			hasCwdTools: () => true,
			hasPrompts: () => true,
			hasSkills: () => false,
			showModelSelector: true
		});

		expect(commands.find((c) => c.action === ChatFormCommandAction.SKILLS)?.disabled).toBe(true);
	});
});

describe('activateSkillByName (explicit /skills <name>)', () => {
	it('resolves the base read through the server and routes it through the shared durable operation', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));

		const outcome = await activateSkillByName('conv-1', 'demo-skill', { cwd: '/repo' });

		expect(outcome).toEqual({ created: true, ok: true });
		expect(mockRead).toHaveBeenCalledWith({ name: 'demo-skill' }, '/repo', undefined);
		expect(mockRecordActivation).toHaveBeenCalledWith({
			conversationId: 'conv-1',
			cwd: '/repo',
			result: baseResult('demo-skill')
		});
	});

	it('reconstructs the durable activation cache before recording so reloads dedupe', async () => {
		mockRead.mockResolvedValue(baseResult('demo-skill'));

		await activateSkillByName('conv-1', 'demo-skill');

		expect(mockLoadConversation).toHaveBeenCalledWith('conv-1');
		expect(mockLoadConversation.mock.invocationCallOrder[0]).toBeLessThan(
			mockRecordActivation.mock.invocationCallOrder[0]
		);
	});

	it('persists nothing when the base read is unavailable or fails', async () => {
		mockRead.mockRejectedValue(new Error('skills disabled'));

		const outcome = await activateSkillByName('conv-1', 'demo-skill');

		expect(outcome).toEqual({ ok: false, reason: 'unavailable' });
		expect(mockRecordActivation).not.toHaveBeenCalled();
		expect(mockLoadConversation).not.toHaveBeenCalled();
	});

	it('persists nothing when the resolved read is not a base skill result', async () => {
		mockRead.mockResolvedValue({
			kind: 'resource',
			skill: { id: 'opaque-demo-skill', name: 'demo-skill', scope: 'project', provider: 'agents' },
			resource: { path: 'refs/DETAILS.md' },
			content_xml: '<skill_resource>data</skill_resource>',
			diagnostics: []
		});

		const outcome = await activateSkillByName('conv-1', 'demo-skill');

		expect(outcome).toEqual({ ok: false, reason: 'not-found' });
		expect(mockRecordActivation).not.toHaveBeenCalled();
		expect(mockLoadConversation).not.toHaveBeenCalled();
	});
});
