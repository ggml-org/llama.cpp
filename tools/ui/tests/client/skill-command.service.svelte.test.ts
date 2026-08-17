// Guards read-first `/skills <name>` activation, creation, deduplication, and errors.

import { ChatFormCommandAction } from '$lib/enums';
import { dispatchSkillActivation } from '$lib/services/skill-command.service';
import { AttachmentType } from '$lib/enums';
import { SkillsService } from '$lib/services/skills.service';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { skillActivationStore } from '$lib/stores/skill-activation.svelte';
import { skillAvailabilityStore } from '$lib/stores/skill-availability.svelte';
import type { SkillBaseReadResult } from '$lib/types/skills';
import { getChatCommands } from '$lib/utils';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/skills.service', () => ({
	SkillsService: { read: vi.fn() }
}));
vi.mock('$lib/stores/conversations.svelte', () => ({
	conversationsStore: {
		activeConversation: null as { cwd: string | null; id: string } | null,
		createConversation: vi.fn(async () => 'conv-1'),
		pendingCwd: null as string | null
	}
}));
vi.mock('$lib/stores/skill-activation.svelte', () => ({
	skillActivationStore: {
		loadConversation: vi.fn(async () => undefined),
		recordActivation: vi.fn(async () => ({ created: true, extra: {}, toolResultMessage: null }))
	}
}));
vi.mock('$lib/stores/skill-availability.svelte', () => ({
	skillAvailabilityStore: { isDisabled: vi.fn(() => false) }
}));

function baseResult(name = 'frontend-design'): SkillBaseReadResult {
	return {
		body_markdown: '# Frontend design',
		content_xml: '<skill />',
		diagnostics: [],
		kind: 'skill',
		resources: { paths: [], truncated: false },
		skill: { id: `opaque-${name}`, name, provider: 'agents', scope: 'global' },
		source: '# Frontend design'
	};
}

beforeEach(() => {
	vi.mocked(SkillsService.read).mockReset();
	vi.mocked(conversationsStore.createConversation).mockClear();
	vi.mocked(skillActivationStore.loadConversation).mockClear();
	vi.mocked(skillActivationStore.recordActivation).mockClear();
	vi.mocked(skillAvailabilityStore.isDisabled).mockReset();
	vi.mocked(skillAvailabilityStore.isDisabled).mockReturnValue(false);
	conversationsStore.activeConversation = null;
	conversationsStore.pendingCwd = null;
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

describe('dispatchSkillActivation', () => {
	it('creates a Skill-named conversation in fresh state and persists activation', async () => {
		vi.mocked(SkillsService.read).mockResolvedValue(baseResult());

		const outcome = await dispatchSkillActivation('frontend-design');

		expect(conversationsStore.createConversation).toHaveBeenCalledWith('Skill: frontend-design');
		expect(skillActivationStore.loadConversation).toHaveBeenCalledWith('conv-1');
		expect(skillActivationStore.recordActivation).toHaveBeenCalledWith(
			expect.objectContaining({ conversationId: 'conv-1' })
		);
		expect(outcome).toEqual({ created: true, ok: true });
	});

	it('uses the active conversation and never creates a new one', async () => {
		conversationsStore.activeConversation = { cwd: '/work', id: 'conv-active' };
		vi.mocked(SkillsService.read).mockResolvedValue(baseResult());

		const outcome = await dispatchSkillActivation('frontend-design');

		expect(conversationsStore.createConversation).not.toHaveBeenCalled();
		expect(SkillsService.read).toHaveBeenCalledWith(
			{ name: 'frontend-design' },
			'/work',
			undefined
		);
		expect(skillActivationStore.recordActivation).toHaveBeenCalledWith(
			expect.objectContaining({ conversationId: 'conv-active' })
		);
		expect(outcome).toEqual({ created: true, ok: true });
	});

	it('reports created: false on an already-activated identity', async () => {
		vi.mocked(SkillsService.read).mockResolvedValue(baseResult());
		vi.mocked(skillActivationStore.recordActivation).mockResolvedValue({
			created: false,
			extra: {
				kind: 'base',
				name: 'frontend-design',
				provider: 'agents',
				scope: 'global',
				skillId: 'opaque-frontend-design',
				state: 'approved',
				type: AttachmentType.SKILL
			},
			toolResultMessage: null
		});

		const outcome = await dispatchSkillActivation('frontend-design');

		expect(outcome).toEqual({ created: false, ok: true });
	});

	it('threads the pending CWD into the read and the recorded activation', async () => {
		conversationsStore.pendingCwd = '/pending';
		vi.mocked(SkillsService.read).mockResolvedValue(baseResult());

		await dispatchSkillActivation('frontend-design');

		expect(SkillsService.read).toHaveBeenCalledWith(
			{ name: 'frontend-design' },
			'/pending',
			undefined
		);
		expect(skillActivationStore.recordActivation).toHaveBeenCalledWith(
			expect.objectContaining({ cwd: '/pending' })
		);
	});

	it('a failed read is unavailable and creates no conversation', async () => {
		vi.mocked(SkillsService.read).mockRejectedValue(new Error('boom'));

		const outcome = await dispatchSkillActivation('frontend-design');

		expect(outcome).toEqual({ ok: false, reason: 'unavailable' });
		expect(conversationsStore.createConversation).not.toHaveBeenCalled();
		expect(skillActivationStore.recordActivation).not.toHaveBeenCalled();
	});

	it('a non-base read is not-found and creates no conversation', async () => {
		vi.mocked(SkillsService.read).mockResolvedValue({
			content_xml: '',
			diagnostics: [],
			kind: 'resource',
			resource: { path: 'x' },
			source: '',
			skill: { id: 'opaque-x', name: 'x', provider: 'agents', scope: 'global' }
		});

		const outcome = await dispatchSkillActivation('frontend-design');

		expect(outcome).toEqual({ ok: false, reason: 'not-found' });
		expect(conversationsStore.createConversation).not.toHaveBeenCalled();
		expect(skillActivationStore.recordActivation).not.toHaveBeenCalled();
	});

	it('rejects a resolved disabled identity before creating or persisting', async () => {
		vi.mocked(SkillsService.read).mockResolvedValue(baseResult());
		vi.mocked(skillAvailabilityStore.isDisabled).mockImplementation(
			(id) => id === 'opaque-frontend-design'
		);

		const outcome = await dispatchSkillActivation('frontend-design');

		expect(SkillsService.read).toHaveBeenCalled();
		expect(outcome).toEqual({ ok: false, reason: 'disabled' });
		expect(conversationsStore.createConversation).not.toHaveBeenCalled();
		expect(skillActivationStore.loadConversation).not.toHaveBeenCalled();
		expect(skillActivationStore.recordActivation).not.toHaveBeenCalled();
	});

	it('a failed persistence after a successful read is persistence-failed and keeps the conversation', async () => {
		vi.mocked(SkillsService.read).mockResolvedValue(baseResult());
		vi.mocked(skillActivationStore.recordActivation).mockRejectedValue(new Error('db full'));

		const outcome = await dispatchSkillActivation('frontend-design');

		expect(outcome).toEqual({ ok: false, reason: 'persistence-failed' });
		expect(conversationsStore.createConversation).toHaveBeenCalledWith('Skill: frontend-design');
	});
});
