// Guards what the reader approves: the trace row of a shell call may show the
// model's own description of it, so the permission card must show the command
// that would actually run, and the row must keep the command beside the
// description.

import ChatMessageActionCardPermissionRequest from '#lib/components/app/chat/ChatMessages/ChatMessageActions/ChatMessageActionCard/ChatMessageActionCardPermissionRequest.svelte';
import ChatMessageAgentTrace from '#lib/components/app/chat/ChatMessages/ChatMessageAgentTrace.svelte';
import { BuiltInTool, MessageRole, MessageType } from '#lib/enums/index.ts';
import type { DatabaseMessage } from '#lib/types/index.ts';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const COMMAND = 'rm -rf ./build';
const DESCRIPTION = 'List files';

function shellCall(): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: 'conv-1',
		id: 'assistant-1',
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		toolCalls: JSON.stringify([
			{
				function: {
					arguments: JSON.stringify({ command: COMMAND, description: DESCRIPTION }),
					name: BuiltInTool.SERVER_EXEC_SHELL_COMMAND
				},
				id: 'c1'
			}
		]),
		type: MessageType.TEXT
	};
}

describe('approving a shell command', () => {
	it('shows the command in the permission card', async () => {
		const { container } = render(ChatMessageActionCardPermissionRequest, {
			command: COMMAND,
			onDecision: () => {},
			serverLabel: 'Server Tools',
			toolName: BuiltInTool.SERVER_EXEC_SHELL_COMMAND
		});

		await tick();

		expect(container.querySelector('pre')?.textContent).toBe(COMMAND);
	});

	it('keeps the command beside a model description in the collapsed row', async () => {
		const { container } = render(ChatMessageAgentTrace, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: shellCall()
		});

		await tick();

		const title = container.querySelector('[data-slot="collapsible-trigger"]')?.textContent ?? '';

		expect(title).toContain(DESCRIPTION);
		expect(title).toContain(COMMAND);
	});
});
