import { describe, expect, it } from 'vitest';
import { parseQuestionToolResult, parseTodoWriteResult } from '$lib/utils';

describe('parseQuestionToolResult', () => {
	it('parses answered question summaries', () => {
		expect(
			parseQuestionToolResult(
				'User has answered your questions: "Pick one"="A", "Why"="Because". You can now continue with the user\'s answers in mind.'
			)
		).toEqual([
			{ question: 'Pick one', answer: 'A' },
			{ question: 'Why', answer: 'Because' }
		]);
	});

	it('returns an empty list for unrelated text', () => {
		expect(parseQuestionToolResult('No question summary here')).toEqual([]);
	});
});

describe('parseTodoWriteResult', () => {
	it('parses todo snapshots from JSON text', () => {
		expect(
			parseTodoWriteResult(`[
				{"content":"Inspect files","status":"completed"},
				{"content":"Patch server","status":"in_progress"}
			]`)
		).toEqual([
			{ content: 'Inspect files', status: 'completed' },
			{ content: 'Patch server', status: 'in_progress' }
		]);
	});

	it('returns an empty list for invalid payloads', () => {
		expect(parseTodoWriteResult('{"oops":true}')).toEqual([]);
		expect(parseTodoWriteResult('not json')).toEqual([]);
	});
});
