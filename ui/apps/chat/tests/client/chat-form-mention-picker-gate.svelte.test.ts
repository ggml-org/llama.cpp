// Guards the @-mention picker's file_glob_search gate: when the server
// does not expose the tool (started without --tools) or the user disabled
// it, the picker still opens but explains why instead of firing searches
// that would only fail with "Search failed".

import ChatFormPickerMention from '#lib/components/app/chat/ChatForm/ChatFormPickers/ChatFormPickerMention.svelte';
import { DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY } from '#lib/constants/index.ts';
import { BuiltInTool } from '#lib/enums/index.ts';
import { ToolsService } from '#lib/services/tools.service.ts';
import { toolsStore } from '#lib/stores/tools.svelte.ts';
import type { OpenAIToolDefinition } from '#lib/types/index.ts';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

const FILE_SEARCH_DEF: OpenAIToolDefinition = {
	function: { description: '', name: BuiltInTool.SERVER_FILE_GLOB_SEARCH, parameters: {} },
	type: 'function'
};
const FILE_SEARCH_KEY = `server:${BuiltInTool.SERVER_FILE_GLOB_SEARCH}`;

// The store keeps its server tool list private; tests inject it through
// the reactive field so the derived gates recompute.
function setServerTools(defs: OpenAIToolDefinition[]) {
	(toolsStore as unknown as { _serverTools: OpenAIToolDefinition[] })._serverTools = defs;
}

function renderPicker() {
	return render(ChatFormPickerMention, {
		isOpen: true,
		onClose: () => {},
		onSelect: () => {},
		query: 'main'
	});
}

afterEach(() => {
	vi.restoreAllMocks();
	setServerTools([]);
	toolsStore.setToolEnabled(FILE_SEARCH_KEY, true);
	localStorage.removeItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY);
});

describe('ChatFormPickerMention file_glob_search gate', () => {
	it('explains that file search is unavailable when the server has no tools', async () => {
		setServerTools([]);
		renderPicker();
		await tick();

		expect(document.body.textContent).toContain(
			'File search is unavailable on this server (started without --tools)'
		);
	});

	it('explains that file search must be enabled when the user disabled it', async () => {
		setServerTools([FILE_SEARCH_DEF]);
		toolsStore.setToolEnabled(FILE_SEARCH_KEY, false);
		renderPicker();
		await tick();

		expect(document.body.textContent).toContain(
			'File search is disabled - enable "Search files" in Settings > Tools to use @-mentions'
		);
	});
});

// a mention holds spaces, so a query with one that found nothing is the message going on
describe('ChatFormPickerMention Enter', () => {
	const enter = () => new KeyboardEvent('keydown', { cancelable: true, key: 'Enter' });

	it('lets Enter send a sentence that matched nothing, and closes', async () => {
		const onClose = vi.fn();
		const screen = render(ChatFormPickerMention, {
			isOpen: true,
			onClose,
			onSelect: () => {},
			query: 'alice about it'
		});

		await tick();

		expect(screen.component.handleKeydown(enter())).toBe(false);
		expect(onClose).toHaveBeenCalled();
	});

	it('keeps Enter for a single word', async () => {
		const screen = renderPicker();

		await tick();

		const event = enter();

		expect(screen.component.handleKeydown(event)).toBe(true);
		expect(event.defaultPrevented).toBe(true);
	});

	it("does not pick the first word's results for the words typed after it", async () => {
		setServerTools([FILE_SEARCH_DEF]);
		// the glob spells each letter as a class (`[tT][oO]...`): the upper-case one tells the word
		vi.spyOn(ToolsService, 'executeToolRaw').mockImplementation(async (_tool, params) => ({
			base: '/home/demo',
			entries: /T.*O.*D.*A.*Y/.test(String((params as { include: string }).include))
				? []
				: [{ path: 'main.ts', type: 'file' }]
		}));

		const onSelect = vi.fn();
		const onClose = vi.fn();
		const screen = render(ChatFormPickerMention, {
			isOpen: true,
			onClose,
			onSelect,
			query: 'main'
		});

		await expect.element(screen.getByText('main.ts').first()).toBeVisible();
		await screen.rerender({ query: 'main today' });

		// the longer query has not been searched yet: Enter neither picks main nor sends
		expect(screen.component.handleKeydown(enter())).toBe(true);
		expect(onSelect).not.toHaveBeenCalled();

		// once it found nothing, Enter sends the message
		await expect.poll(() => screen.component.handleKeydown(enter())).toBe(false);
		expect(onSelect).not.toHaveBeenCalled();
		expect(onClose).toHaveBeenCalled();
	});
});
