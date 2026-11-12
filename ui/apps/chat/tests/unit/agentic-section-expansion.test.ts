// Truth table for the default expansion of agentic sections - what a tool call
// block does before the user touches it.
//
// The interesting row is a call the agentic loop reports as running while its
// section is already TOOL_CALL: exec_shell_command persists an empty result
// before any output is read, so the section type alone cannot tell a live call
// from a finished one.

import { AgenticSectionType } from '$lib/enums';
import type { AgenticSection } from '$lib/types';
import { isSectionExpandedByDefault } from '$lib/utils';
import { describe, expect, it } from 'vitest';

const CALL_ID = 'call-1';

function section(overrides: Partial<AgenticSection> = {}): AgenticSection {
	return { content: '', type: AgenticSectionType.TOOL_CALL, ...overrides };
}

function expanded(section: AgenticSection, flags: Record<string, unknown> = {}): boolean {
	return isSectionExpandedByDefault(section, {
		alwaysShowToolCallContent: false,
		executingToolCallId: null,
		showThoughtInProgress: false,
		showToolCallInProgress: false,
		...flags
	});
}

describe('isSectionExpandedByDefault', () => {
	it('expands a call whose arguments are streaming when the setting is on', () => {
		const streaming = section({
			toolCallId: CALL_ID,
			toolName: 'read_file',
			type: AgenticSectionType.TOOL_CALL_STREAMING
		});

		expect(expanded(streaming, { showToolCallInProgress: true })).toBe(true);
		expect(expanded(streaming)).toBe(false);
	});

	it('expands a call waiting for its result when the setting is on', () => {
		const pending = section({ toolCallId: CALL_ID, type: AgenticSectionType.TOOL_CALL_PENDING });

		expect(expanded(pending, { showToolCallInProgress: true })).toBe(true);
		expect(expanded(pending)).toBe(false);
	});

	it('collapses a finished call unless "always show" is on', () => {
		const finished = section({ toolCallId: CALL_ID, toolResult: 'output' });

		expect(expanded(finished, { showToolCallInProgress: true })).toBe(false);
		expect(expanded(finished, { alwaysShowToolCallContent: true })).toBe(true);
	});

	it('keeps the running call expanded even when its result is already persisted', () => {
		// exec_shell_command: empty result created, output still streaming
		const live = section({ toolCallId: CALL_ID, toolResult: '' });

		expect(expanded(live, { executingToolCallId: CALL_ID, showToolCallInProgress: true })).toBe(
			true
		);
	});

	it('does not treat another call finishing as the running one', () => {
		const other = section({ toolCallId: 'call-2', toolResult: 'output' });

		expect(expanded(other, { executingToolCallId: CALL_ID, showToolCallInProgress: true })).toBe(
			false
		);
	});

	it('still leaves the running call collapsed when both settings are off', () => {
		const live = section({ toolCallId: CALL_ID, toolResult: '' });

		expect(expanded(live, { executingToolCallId: CALL_ID })).toBe(false);
	});

	it('never reads a missing executing id as a running call', () => {
		expect(expanded(section({ toolCallId: CALL_ID }), { executingToolCallId: null })).toBe(false);
		expect(expanded(section({ type: AgenticSectionType.TEXT }), { executingToolCallId: '' })).toBe(
			false
		);
	});

	it('expands a running call when "always show" is on and the in-progress one is off', () => {
		const pending = section({ toolCallId: CALL_ID, type: AgenticSectionType.TOOL_CALL_PENDING });

		expect(expanded(pending, { alwaysShowToolCallContent: true })).toBe(true);
	});

	it('follows the thought setting for reasoning and never for finished thoughts', () => {
		expect(
			expanded(
				{ content: 'think', type: AgenticSectionType.REASONING_PENDING },
				{ showThoughtInProgress: true }
			)
		).toBe(true);
		expect(
			expanded(
				{ content: 'think', type: AgenticSectionType.REASONING },
				{
					showThoughtInProgress: true
				}
			)
		).toBe(false);
	});

	it('never expands plain text', () => {
		expect(
			expanded(
				{ content: 'hello', type: AgenticSectionType.TEXT },
				{
					alwaysShowToolCallContent: true,
					executingToolCallId: CALL_ID,
					showThoughtInProgress: true,
					showToolCallInProgress: true
				}
			)
		).toBe(false);
	});
});
