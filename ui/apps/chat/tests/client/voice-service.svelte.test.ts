// Guards the wiring every voice feature starts its session with: the Voice settings become
// capture options (boost, processing, a caller's own constraints), the language and the
// transcription model go where a router needs them, and an API key keeps the realtime
// socket off, since a browser websocket can not send it, which the app says of a model that
// only transcribes over that socket.

import { seedStores } from '../stories/chat/form/fixtures';
import { VOICE_MICROPHONE_DEFAULT, VoiceTranscriptionError } from '@ggml-org/llama-voice';
import { SETTINGS_KEYS, VOICE_MODE_MESSAGE } from '#lib/constants/index.ts';
import { SETTING_CONFIG_DEFAULT } from '#lib/constants/settings.constants.ts';
import { VoiceService } from '#lib/services/voice.service.ts';
import type { SettingsConfigType } from '#lib/types/index.ts';
import { afterEach, describe, expect, it } from 'vitest';

let restore: (() => void) | undefined;

afterEach(() => {
	restore?.();
	restore = undefined;
});

/** The default settings with `changes`, on the system microphone so no device is listed. */
function settings(changes: Record<string, unknown> = {}): SettingsConfigType {
	return {
		...SETTING_CONFIG_DEFAULT,
		[SETTINGS_KEYS.VOICE_MICROPHONE]: VOICE_MICROPHONE_DEFAULT,
		...changes
	} as SettingsConfigType;
}

describe('VoiceService.sessionOptions', () => {
	it('boosts the raw signal by default, and transcribes with the model a router routes on', async () => {
		const options = await VoiceService.sessionOptions(settings(), {
			isRouter: true,
			model: 'org/hearing-model'
		});

		expect(options.capture).toEqual({ boost: true });
		expect(options.transcribe).toMatchObject({ language: undefined, model: 'org/hearing-model' });
	});

	it('leaves the model out for a single-model server', async () => {
		const options = await VoiceService.sessionOptions(settings(), {
			isRouter: false,
			model: 'org/hearing-model'
		});

		expect(options.transcribe?.model).toBeUndefined();
	});

	it('follows the processing, boost and language settings', async () => {
		const options = await VoiceService.sessionOptions(
			settings({
				[SETTINGS_KEYS.VOICE_AUTO_GAIN]: true,
				[SETTINGS_KEYS.VOICE_BOOST]: false,
				[SETTINGS_KEYS.VOICE_NOISE_SUPPRESSION]: true,
				[SETTINGS_KEYS.VOICE_TRANSCRIPTION_LANGUAGE]: ' de '
			}),
			{ isRouter: false, model: null }
		);

		expect(options.capture).toEqual({
			boost: false,
			constraints: { autoGainControl: true, noiseSuppression: true }
		});
		expect(options.transcribe?.language).toBe('de');
	});

	it("takes a caller's own constraints over the processing settings", async () => {
		const options = await VoiceService.sessionOptions(
			settings({ [SETTINGS_KEYS.VOICE_NOISE_SUPPRESSION]: true }),
			{ constraints: { echoCancellation: true }, isRouter: false, model: null }
		);

		expect(options.capture).toMatchObject({ constraints: { echoCancellation: true } });
	});

	it('records without transcribing when asked to', async () => {
		const options = await VoiceService.sessionOptions(settings(), {
			isRouter: false,
			model: null,
			transcribe: false
		});

		expect(options.transcribe).toBeUndefined();
	});
});

describe('VoiceService.serverContext', () => {
	it('uses the realtime socket when no API key is set', () => {
		restore = seedStores({ config: { apiKey: '' } });

		const { context, realtime } = VoiceService.serverContext();

		expect(realtime).toBe(true);
		expect(context.getHeaders?.()).toEqual({});
		expect(context.resolveUrl('./v1/realtime')).toMatch(/\/v1\/realtime$/);
	});

	it('keeps to HTTP, with the key, when the server needs one', () => {
		restore = seedStores({ config: { apiKey: 'secret' } });

		const { context, realtime } = VoiceService.serverContext();

		expect(realtime).toBe(false);
		expect(context.getHeaders?.()).toEqual({ Authorization: 'Bearer secret' });
	});
});

describe('VoiceService.failureReason', () => {
	const notSupported = new VoiceTranscriptionError(
		'this server does not support chat/completions',
		501
	);

	it('says a live-only model cannot transcribe on a server with an API key', () => {
		restore = seedStores({ config: { apiKey: 'secret' } });

		expect(VoiceService.failureReason(notSupported)).toBe(VOICE_MODE_MESSAGE.realtimeOnlyBehindKey);
	});

	it("leaves other failures to the server's own words", () => {
		restore = seedStores({ config: { apiKey: 'secret' } });

		expect(
			VoiceService.failureReason(new VoiceTranscriptionError('Loading model', 503))
		).toBeNull();
		expect(VoiceService.failureReason(new Error('Failed to fetch'))).toBeNull();

		restore();
		restore = seedStores({ config: { apiKey: '' } });

		// without a key the realtime socket was tried first: the model simply cannot hear
		expect(VoiceService.failureReason(notSupported)).toBeNull();
	});
});
