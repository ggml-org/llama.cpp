import {
	VOICE_ASR_TEXT_PREFIX_PATTERN,
	VOICE_ASR_TEXT_TAG_PATTERN,
	VOICE_SEGMENTS_CONTENT_PATTERN,
	VOICE_SEGMENTS_START_PATTERN,
	VOICE_UNFINISHED_ESCAPE_PATTERN
} from './constants.js';

/**
 * A transcript as the speaker's words alone. Some speech-to-text models wrap them in a
 * format of their own, which the server passes on as it is: Qwen3-ASR names the language
 * first (`language French<asr_text>Bonjour`), VibeVoice-ASR answers with timed segments
 * (`[{"Start":0,"End":3.1,"Speaker":0,"Content":"Bonjour"}]`). Text in neither format is
 * returned untouched, so text still streaming keeps its spaces.
 *
 * It takes text that is still streaming too: segments cut short give the words they
 * hold so far, and nothing before the first one.
 */
export function cleanTranscript(text: string): string {
	if (VOICE_SEGMENTS_START_PATTERN.test(text)) return segmentsText(text);

	// search() keeps no position in the shared pattern
	if (text.search(VOICE_ASR_TEXT_TAG_PATTERN) === -1) return text;

	return text
		.replace(VOICE_ASR_TEXT_PREFIX_PATTERN, '')
		.replace(VOICE_ASR_TEXT_TAG_PATTERN, '')
		.trim();
}

/** The words of timed segments, in order, whether or not the JSON is complete. */
function segmentsText(text: string): string {
	const contents: string[] = [];

	for (const match of text.matchAll(VOICE_SEGMENTS_CONTENT_PATTERN)) {
		const content = unescapeJsonString(match[1]).trim();

		if (content) contents.push(content);
	}

	return contents.join(' ');
}

/** The value of a JSON string's inside, which may stop in the middle of an escape. */
function unescapeJsonString(inside: string): string {
	try {
		return JSON.parse(`"${inside}"`);
	} catch {
		try {
			return JSON.parse(`"${inside.replace(VOICE_UNFINISHED_ESCAPE_PATTERN, '')}"`);
		} catch {
			return inside;
		}
	}
}
