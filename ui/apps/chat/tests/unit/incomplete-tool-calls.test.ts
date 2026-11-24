import { MessageRole } from '#lib/enums/index.ts';
import type { ApiChatCompletionToolCall, ApiChatMessageData } from '#lib/types/api.d.ts';
import { dropIncompleteToolCalls } from '#lib/utils/incomplete-tool-calls.ts';
import { describe, expect, it } from 'vitest';

function call(id: string, args: string, name = 'exec_shell_command'): ApiChatCompletionToolCall {
	return { function: { arguments: args, name }, id, type: 'function' };
}

const CUT = '{"command": "cd /repo && cat > /tmp/x.mjs <<\'EOF\'\\nimport';

describe('dropIncompleteToolCalls', () => {
	it('leaves complete tool calls and their results alone', () => {
		const messages: ApiChatMessageData[] = [
			{ content: 'run it', role: MessageRole.USER },
			{ content: '', role: MessageRole.ASSISTANT, tool_calls: [call('a', '{"command":"ls"}')] },
			{ content: 'file.txt', role: MessageRole.TOOL, tool_call_id: 'a' }
		];
		const result = dropIncompleteToolCalls(messages);

		expect(result).toEqual(messages);
		expect(result[1]).toBe(messages[1]);
	});

	it('drops a call whose arguments were cut off, keeping the text of the message', () => {
		const result = dropIncompleteToolCalls([
			{
				content: 'Let me write the script.',
				role: MessageRole.ASSISTANT,
				tool_calls: [call('a', CUT)]
			},
			{ content: 'Continue', role: MessageRole.USER }
		]);

		expect(result).toEqual([
			{ content: 'Let me write the script.', role: MessageRole.ASSISTANT },
			{ content: 'Continue', role: MessageRole.USER }
		]);
	});

	it('keeps the complete calls of a message and drops the results of the cut one', () => {
		const result = dropIncompleteToolCalls([
			{
				content: '',
				role: MessageRole.ASSISTANT,
				tool_calls: [call('a', '{"command":"ls"}'), call('b', CUT)]
			},
			{ content: 'file.txt', role: MessageRole.TOOL, tool_call_id: 'a' },
			{ content: 'error', role: MessageRole.TOOL, tool_call_id: 'b' }
		]);

		expect(result).toEqual([
			{ content: '', role: MessageRole.ASSISTANT, tool_calls: [call('a', '{"command":"ls"}')] },
			{ content: 'file.txt', role: MessageRole.TOOL, tool_call_id: 'a' }
		]);
	});

	it('drops a call with no name and sends a call with no arguments as {}', () => {
		const result = dropIncompleteToolCalls([
			{
				content: '',
				role: MessageRole.ASSISTANT,
				tool_calls: [call('a', '{}', ''), call('b', '', 'list_tasks')]
			}
		]);

		expect(result[0].tool_calls).toEqual([call('b', '{}', 'list_tasks')]);
	});
});
