import { describe, expect, it } from 'vitest';
import { parseQuestionToolResult } from '$lib/utils';

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
