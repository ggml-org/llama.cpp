// Guards how a voice message goes to a model: the audio alone to a model with audio input,
// its transcript as text to one without, and the copied message carrying the transcript.

import { AttachmentType, ContentPartType, MessageRole, MessageType } from '#lib/enums/index.ts';
import { ChatService } from '#lib/services/chat.service.ts';
import { modelsStore } from '#lib/stores/models/index.svelte.ts';
import { withTranscript } from '#lib/stores/voice-transcripts.svelte.ts';
import type { DatabaseMessage, DatabaseMessageExtra } from '#lib/types/index.ts';
import { formatMessageForClipboard } from '#lib/utils/clipboard.ts';
import { afterEach, describe, expect, it, vi } from 'vitest';

const voiceClip: DatabaseMessageExtra = {
	base64Data: 'UklGRg==',
	durationMs: 2400,
	mimeType: 'audio/wav',
	name: 'voice-1.wav',
	peaks: [0.2, 1, 0.4],
	transcript: 'What is the weather like on Mars?',
	type: AttachmentType.AUDIO,
	voice: true
};

function userMessage(extra: DatabaseMessageExtra[], content = ''): DatabaseMessage {
	return {
		children: [],
		content,
		convId: 'c1',
		extra,
		id: 'm1',
		parent: null,
		role: MessageRole.USER,
		timestamp: 0,
		type: MessageType.TEXT
	} as DatabaseMessage;
}

afterEach(() => {
	vi.restoreAllMocks();
});

describe('sending a voice message', () => {
	it('sends the audio alone to a model with audio input', async () => {
		const result = await ChatService.convertDbMessageToApiChatMessageData(
			userMessage([voiceClip]),
			{ audioInput: true }
		);
		const parts = result.content as { type: string; text?: string }[];

		expect(parts).toHaveLength(1);
		expect(parts[0].type).toBe(ContentPartType.INPUT_AUDIO);
		expect(JSON.stringify(parts)).not.toContain('Mars');
	});

	it('sends the transcript as text to a model without audio input', async () => {
		const result = await ChatService.convertDbMessageToApiChatMessageData(
			userMessage([voiceClip], 'Answer briefly.'),
			{ audioInput: false }
		);
		const parts = result.content as { type: string; text?: string }[];

		expect(parts.map((part) => part.type)).toEqual([ContentPartType.TEXT, ContentPartType.TEXT]);
		expect(parts[0].text).toContain('Voice message transcript');
		expect(parts[0].text).toContain('What is the weather like on Mars?');
		expect(parts[1].text).toBe('Answer briefly.');
	});

	it('keeps the audio when there is no transcript to send instead', async () => {
		const { transcript: _transcript, ...withoutTranscript } = voiceClip as {
			transcript?: string;
		} & DatabaseMessageExtra;
		const result = await ChatService.convertDbMessageToApiChatMessageData(
			userMessage([withoutTranscript as DatabaseMessageExtra]),
			{ audioInput: false }
		);

		expect((result.content as { type: string }[])[0].type).toBe(ContentPartType.INPUT_AUDIO);
	});

	it('keeps the audio while the model modalities are unknown', () => {
		vi.spyOn(modelsStore.props, 'getModelModalities').mockReturnValue(null);

		expect(ChatService.conversionOptionsFor('some-model')).toEqual({});
	});

	it('reads the audio input from the model modalities', () => {
		vi.spyOn(modelsStore.props, 'getModelModalities').mockReturnValue({
			audio: false,
			video: false,
			vision: true
		});

		expect(ChatService.conversionOptionsFor('text-model')).toEqual({
			audioInput: false,
			visionInput: true
		});
	});
});

describe('a transcript that arrives after the send', () => {
	it('fills in the clip it belongs to only', () => {
		const pending = { ...voiceClip, transcript: undefined } as DatabaseMessageExtra;
		const other = { ...pending, name: 'voice-2.wav' } as DatabaseMessageExtra;
		const next = withTranscript([pending, other], 'voice-1.wav', 'Hello');

		expect(next?.[0]).toMatchObject({ name: 'voice-1.wav', transcript: 'Hello' });
		expect(next?.[1]).toBe(other);
	});

	it('leaves a message without that clip alone', () => {
		expect(withTranscript([voiceClip], 'voice-1.wav', 'Hello')).toBeNull();
		expect(withTranscript(undefined, 'voice-1.wav', 'Hello')).toBeNull();
	});
});

describe('copying a voice message', () => {
	it('copies the transcript as the text', () => {
		expect(formatMessageForClipboard('', [voiceClip])).toBe('What is the weather like on Mars?');
		expect(formatMessageForClipboard('Briefly.', [voiceClip], true)).toBe(
			'What is the weather like on Mars?\n\nBriefly.'
		);
	});
});
