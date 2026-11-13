// Settings > Tools must edit the active conversation policy, not the defaults for future chats.

import SettingsChatToolsTab from '$lib/components/app/settings/SettingsChat/SettingsChatToolsTab.svelte';
import { DatabaseService } from '$lib/services/database.service';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type { DatabaseConversation, OpenAIToolDefinition } from '$lib/types';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

const TOOL_KEY = 'server:test_tool';
const TOOL: OpenAIToolDefinition = {
	function: { description: '', name: 'test_tool', parameters: {} },
	type: 'function'
};

function setServerTools(definitions: OpenAIToolDefinition[]) {
	(toolsStore as unknown as { _serverTools: OpenAIToolDefinition[] })._serverTools = definitions;
}

afterEach(() => {
	conversationsStore.clearActiveConversation();
	conversationsStore.conversations = [];
	setServerTools([]);
	toolsStore.setToolEnabled(TOOL_KEY, true);
	vi.restoreAllMocks();
});

describe('SettingsChatToolsTab', () => {
	it('keeps a tool change in the active conversation', async () => {
		vi.spyOn(DatabaseService, 'updateConversation').mockResolvedValue(undefined);
		const conversation: DatabaseConversation = {
			currNode: null,
			id: 'conversation-1',
			lastModified: 0,
			name: 'Conversation'
		};

		conversationsStore.activeConversation = conversation;
		conversationsStore.conversations = [conversation];
		setServerTools([TOOL]);
		const screen = render(SettingsChatToolsTab);
		const trigger = screen.container.querySelector<HTMLElement>('[data-slot="collapsible-trigger"]');

		if (!trigger) throw new Error('tool group trigger not found');

		await trigger.click();
		await tick();

		const checkboxes = screen.container.querySelectorAll<HTMLElement>('[role="checkbox"]');

		await checkboxes[1].click();
		await tick();

		expect(conversationsStore.activeConversation?.disabledTools).toEqual([TOOL_KEY]);
		expect(toolsStore.isToolEnabled(TOOL_KEY)).toBe(true);
	});
});
