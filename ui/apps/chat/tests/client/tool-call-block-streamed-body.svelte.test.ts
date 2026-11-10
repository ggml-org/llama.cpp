// A file tool call must show what it streamed even when it carries no
// path: the path only names the file in the block title, so a call that
// streams its body first (or a same-named tool with other arg names) used
// to render an empty block.

import ChatMessageToolCallBlockEditFile from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockEditFile.svelte';
import ChatMessageToolCallBlockWriteFile from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockWriteFile.svelte';
import { AgenticSectionType, BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/types';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const WRITE_FILE_BODY = 'body streamed without a path';
const EDIT_FILE_NEW_TEXT = 'replacement text';

function makeSection(toolName: BuiltInTool, toolArgs: string): AgenticSection {
	return {
		content: '',
		toolArgs,
		toolName,
		toolResult: undefined,
		type: AgenticSectionType.TOOL_CALL
	};
}

describe('file tool calls render their body without a path', () => {
	it('write_file shows the content it streamed', async () => {
		render(ChatMessageToolCallBlockWriteFile, {
			isStreaming: true,
			open: true,
			section: makeSection(BuiltInTool.SERVER_WRITE_FILE, `{"content":"${WRITE_FILE_BODY}"`)
		});

		await tick();

		expect(document.body.textContent).toContain(WRITE_FILE_BODY);
	});

	it('edit_file shows the diff it streamed', async () => {
		render(ChatMessageToolCallBlockEditFile, {
			isStreaming: true,
			open: true,
			section: makeSection(
				BuiltInTool.SERVER_EDIT_FILE,
				`{"edits":[{"old_text":"stale text","new_text":"${EDIT_FILE_NEW_TEXT}"}]}`
			)
		});

		await tick();

		expect(document.body.textContent).toContain(EDIT_FILE_NEW_TEXT);
		expect(document.body.textContent).toContain('stale text');
	});

	it('write_file falls back to the raw args when no known content key is present', async () => {
		render(ChatMessageToolCallBlockWriteFile, {
			isStreaming: false,
			open: true,
			section: makeSection(BuiltInTool.SERVER_WRITE_FILE, '{"contents":"x"}')
		});

		await tick();

		expect(document.body.textContent).toContain('contents');
	});

	it('write_file renders an empty code block for a finished empty write', async () => {
		// content of '' used to fall through to the raw-args fallback
		const section = makeSection(BuiltInTool.SERVER_WRITE_FILE, '{"content":""}');

		section.toolResult = '{"result":"File written","bytes":0}';

		render(ChatMessageToolCallBlockWriteFile, { isStreaming: false, open: true, section });

		await tick();

		expect(document.querySelector('pre code.hljs')).toBeTruthy();
		expect(document.body.textContent).not.toContain('"content"');
		expect(document.body.textContent).toMatch(/0\s*bytes/);
	});
});
