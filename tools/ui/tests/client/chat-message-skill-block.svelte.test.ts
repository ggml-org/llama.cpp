// Guards the purpose-built Skills result renderer end to end: base
// activations and resource reads render typed labels (title, provider ·
// scope · path detail) while the server XML stays opaque plain text, and
// malformed/unknown records fall back to the generic tool card. The
// routing decision lives in ChatMessageToolCallBlock, so these cases are
// exercised through it — the same entry point the chat message renderer
// uses.

import ChatMessageToolCallBlock from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlock.svelte';
import { AgenticSectionType } from '$lib/enums';
import { AttachmentType } from '$lib/enums';
import type { DatabaseMessageExtraSkill } from '$lib/types';
import type { AgenticSection } from '$lib/utils';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

function skillExtra(overrides: Partial<DatabaseMessageExtraSkill> = {}): DatabaseMessageExtraSkill {
	return {
		type: AttachmentType.SKILL,
		kind: 'base',
		state: 'approved',
		skillId: 'opaque-id-1',
		name: 'add-new-model',
		scope: 'project',
		provider: 'agents',
		...overrides
	};
}

function section(overrides: Partial<AgenticSection> = {}): AgenticSection {
	return {
		type: AgenticSectionType.TOOL_CALL,
		content: '',
		toolName: 'read_skill',
		toolResult: '<skill_content name="add-new-model">body &amp; more</skill_content>',
		...overrides
	};
}

async function renderBlock(sectionData: AgenticSection) {
	const { container } = render(ChatMessageToolCallBlock, {
		isStreaming: false,
		open: true,
		section: sectionData
	});

	await tick();

	return container;
}

function textOf(container: HTMLElement): string {
	return (container.textContent ?? '').replace(/\s+/g, ' ').trim();
}

describe('read_skill result rendering', () => {
	it('renders a base activation with typed labels and opaque XML text', async () => {
		const container = await renderBlock(
			section({ toolResultExtras: [skillExtra()] })
		);

		const text = textOf(container);

		expect(text).toContain('Skill · add-new-model');
		expect(text).toContain('agents · project');
		// The XML is ordinary text content — the entity is preserved as the
		// literal characters it arrived in, never decoded or re-parsed.
		expect(text).toContain('<skill_content name="add-new-model">body &amp; more</skill_content>');
		// The XML is never parsed into DOM markup.
		expect(container.querySelector('skill_content')).toBeNull();
	});

	it('renders a resource read with its relative path in the detail line', async () => {
		const container = await renderBlock(
			section({
				toolResult: '<skill_resource>data</skill_resource>',
				toolResultExtras: [
					skillExtra({ kind: 'resource', path: 'refs/DETAILS.md', metadata: undefined })
				]
			})
		);

		const text = textOf(container);

		expect(text).toContain('Skill resource · add-new-model');
		expect(text).toContain('agents · project · refs/DETAILS.md');
		expect(text).toContain('<skill_resource>data</skill_resource>');
		expect(container.querySelector('skill_resource')).toBeNull();
	});

	it('falls back to the generic tool card for a read_skill section without valid metadata', async () => {
		const container = await renderBlock(section({ toolResultExtras: [] }));

		const text = textOf(container);

		expect(text).toContain('read_skill');
		expect(text).not.toContain('Skill ·');
		expect(text).not.toContain('Skill resource ·');
	});

	it('shows the pending state while the read is in flight', async () => {
		const container = await renderBlock(
			section({ type: AgenticSectionType.TOOL_CALL_PENDING, toolResultExtras: [skillExtra()] })
		);

		expect(textOf(container)).toContain('Waiting for result...');
	});

	it('shows the no-output state for a terminal section without a result', async () => {
		const container = await renderBlock(section({ toolResult: undefined, toolResultExtras: [skillExtra()] }));

		expect(textOf(container)).toContain('No output');
	});
});
