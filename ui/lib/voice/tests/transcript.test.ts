// Guards that a transcript is the speaker's words alone: the formats speech-to-text models
// wrap them in (Qwen3-ASR's language tag, VibeVoice-ASR's timed segments) are taken off,
// also while the text still streams, and plain text is left as it is.

import { cleanTranscript } from '../src/transcript.js';
import { describe, expect, it } from 'vitest';

describe('cleanTranscript', () => {
	it('leaves plain text as it is, spaces included', () => {
		expect(cleanTranscript(' Hello there. ')).toBe(' Hello there. ');
		expect(cleanTranscript('')).toBe('');
		// a bracket or the word "language" alone is not a format
		expect(cleanTranscript('[laughs] language matters')).toBe('[laughs] language matters');
	});

	it('takes off the language and tag Qwen3-ASR opens with', () => {
		expect(cleanTranscript('language French<asr_text>Bonjour à tous.')).toBe('Bonjour à tous.');
		expect(cleanTranscript('<asr_text>Hello.</asr_text>')).toBe('Hello.');
		// silence: the model names no language and says nothing
		expect(cleanTranscript('language None<asr_text>')).toBe('');
	});

	it('cleans the same text again and again', () => {
		// the tag pattern is shared and global: it must keep no position between calls
		for (let i = 0; i < 3; i++) {
			expect(cleanTranscript('language English<asr_text>Hi.')).toBe('Hi.');
		}
	});

	it('joins the words of the timed segments VibeVoice-ASR answers with', () => {
		const segments = JSON.stringify([
			{ Content: 'Bonjour, ça va ?', End: 3.1, Speaker: 0, Start: 0 },
			{ Content: ' Oui, "très" bien.', End: 5.4, Speaker: 1, Start: 3.1 }
		]);

		expect(cleanTranscript(segments)).toBe('Bonjour, ça va ? Oui, "très" bien.');
		expect(cleanTranscript('[{"Start":0.0,"End":1.0,"Speaker":0,"Content":""}]')).toBe('');
	});

	it('shows only the words of segments that are still streaming', () => {
		expect(cleanTranscript('[')).toBe('');
		expect(cleanTranscript('[{"Start":0.0,"End":3.1,"Spe')).toBe('');
		expect(cleanTranscript('[{"Start":0.0,"End":3.1,"Speaker":0,"Content":"Bonjour, ça')).toBe(
			'Bonjour, ça'
		);
		// cut inside an escape: the half of it is left out
		expect(cleanTranscript('[{"Start":0,"Content":"Line one\\')).toBe('Line one');
		expect(cleanTranscript('[{"Start":0,"Content":"caf\\u00')).toBe('caf');
		expect(cleanTranscript('[{"Start":0,"Content":"caf\\u00e9"},{"Start":1,"Content":"oui')).toBe(
			'café oui'
		);
	});
});
