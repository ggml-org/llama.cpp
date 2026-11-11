import { markdownLanguage } from '@codemirror/lang-markdown';
import { LanguageSupport } from '@codemirror/language';
import { EditorState } from '@codemirror/state';
import { codeRanges } from '$lib/components/app/content/MarkdownEditor/markdown-syntax';
import { describe, expect, it } from 'vitest';

function stateOf(doc: string): EditorState {
	return EditorState.create({ doc, extensions: new LanguageSupport(markdownLanguage) });
}

describe('codeRanges', () => {
	it('lists inline code and code blocks in range', () => {
		const doc = 'a `b` c\n\n~~~\nd\n~~~';
		const state = stateOf(doc);

		expect(codeRanges(state, 0, doc.length)).toEqual([
			[2, 5],
			[9, doc.length]
		]);
	});
});
