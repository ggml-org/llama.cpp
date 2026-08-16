import { ChatFormCommandAction } from '$lib/enums';
import { getChatCommands } from '$lib/utils';
import { describe, expect, it } from 'vitest';

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
