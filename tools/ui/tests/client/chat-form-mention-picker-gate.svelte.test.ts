// Guards the @-mention picker's file_glob_search gate: when the server
// does not expose the tool (started without --tools) or the user disabled
// it, the picker still opens but explains why instead of firing searches
// that would only fail with "Search failed".

import ChatFormMentionPicker from '$lib/components/app/chat/ChatForm/ChatFormPickers/ChatFormMentionPicker.svelte';
import { DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import { toolsStore } from '$lib/stores/tools.svelte';
import type { OpenAIToolDefinition } from '$lib/types';
import { tick } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const FILE_SEARCH_DEF: OpenAIToolDefinition = {
	function: { description: '', name: BuiltInTool.FILE_GLOB_SEARCH, parameters: {} },
	type: 'function'
};
const FILE_SEARCH_KEY = `builtin:${BuiltInTool.FILE_GLOB_SEARCH}`;

// The store keeps its builtin tool list private; tests inject it through
// the reactive field so the derived gates recompute.
function setBuiltinTools(defs: OpenAIToolDefinition[]) {
	(toolsStore as unknown as { _builtinTools: OpenAIToolDefinition[] })._builtinTools = defs;
}

function renderPicker() {
	return render(ChatFormMentionPicker, {
		isOpen: true,
		onClose: () => {},
		onSelect: () => {},
		query: 'main'
	});
}

afterEach(() => {
	setBuiltinTools([]);
	toolsStore.setToolEnabled(FILE_SEARCH_KEY, true);
	localStorage.removeItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY);
});

describe('ChatFormMentionPicker file_glob_search gate', () => {
	it('explains that file search is unavailable when the server has no tools', async () => {
		setBuiltinTools([]);
		renderPicker();
		await tick();

		expect(document.body.textContent).toContain(
			'File search is unavailable on this server (started without --tools)'
		);
	});

	it('explains that file search must be enabled when the user disabled it', async () => {
		setBuiltinTools([FILE_SEARCH_DEF]);
		toolsStore.setToolEnabled(FILE_SEARCH_KEY, false);
		renderPicker();
		await tick();

		expect(document.body.textContent).toContain(
			'File search is disabled - enable "Search files" in Settings > Tools to use @-mentions'
		);
	});
});
