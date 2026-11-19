import { SETTINGS_KEYS } from '#lib/constants/index.ts';
import type { SettingsConfigType } from '#lib/types/index.ts';
import { getAuthHeaders } from '#lib/utils/api-headers.ts';
import { withBase } from '#lib/utils/base-path.ts';
import {
	resolveMicrophone,
	VOICE_MICROPHONE_AUTO,
	type VoiceServerContext,
	type VoiceSessionOptions
} from '#lib/voice/index.ts';

/** What a feature asks of the voice wiring, besides the settings. */
export interface VoiceSessionRequest {
	/** The model that transcribes, from resolveVoiceMode(); null leaves it to the server */
	model: string | null;
	/** Router mode: the model goes in the request, as the router routes on it */
	isRouter: boolean;
	/** False records without transcribing, e.g. a voice message when transcripts are off */
	transcribe?: boolean;
	/**
	 * The microphone processing to ask for, in place of what the Voice settings switch on,
	 * e.g. echo cancellation for a spoken conversation that plays replies aloud
	 */
	constraints?: MediaTrackConstraints;
}

/** The session options voice features share: where to capture from, and how to transcribe. */
export type VoiceSessionWiring = Pick<VoiceSessionOptions, 'capture' | 'realtime' | 'transcribe'>;

/**
 * **VoiceService** - How the voice engine (#lib/voice) reaches this app's server and settings
 *
 * The engine imports no app code: it is handed a VoiceServerContext and plain capture
 * options. This builds both from the active server and the Voice settings, once for every
 * feature that records (the chat form's dictation and voice messages, voice mode), so they
 * reach the server the same way and follow the same settings.
 */
export class VoiceService {
	/** The transcription language from the Voice settings, undefined to let the model tell. */
	static language(config: SettingsConfigType): string | undefined {
		return String(config[SETTINGS_KEYS.VOICE_TRANSCRIPTION_LANGUAGE] ?? '').trim() || undefined;
	}

	/**
	 * The microphone processing the Voice settings switch on, laid over the engine's raw
	 * signal; undefined when none is.
	 */
	static processing(config: SettingsConfigType): MediaTrackConstraints | undefined {
		const processing: MediaTrackConstraints = {
			...(config[SETTINGS_KEYS.VOICE_NOISE_SUPPRESSION] === true ? { noiseSuppression: true } : {}),
			...(config[SETTINGS_KEYS.VOICE_ECHO_CANCELLATION] === true ? { echoCancellation: true } : {}),
			...(config[SETTINGS_KEYS.VOICE_AUTO_GAIN] === true ? { autoGainControl: true } : {})
		};

		return Object.keys(processing).length > 0 ? processing : undefined;
	}

	/**
	 * The server's routes and auth for the engine, and whether its realtime socket can be
	 * used: a browser websocket can not send the API key, so a server that needs one gets
	 * HTTP only.
	 */
	static serverContext(): { context: VoiceServerContext; realtime: boolean } {
		const hasAuth = Object.keys(getAuthHeaders()).some(
			(key) => key.toLowerCase() === 'authorization'
		);

		return {
			context: {
				getHeaders: () => getAuthHeaders(),
				resolveUrl: (path) => withBase(path.replace(/^\./, ''))
			},
			realtime: !hasAuth
		};
	}

	/**
	 * The capture, realtime and transcribe options of a voice session, from the Voice
	 * settings (microphone, processing, boost, language) and the active server. Async:
	 * picking the microphone may list the devices.
	 */
	static async sessionOptions(
		config: SettingsConfigType,
		request: VoiceSessionRequest
	): Promise<VoiceSessionWiring> {
		const { context, realtime } = VoiceService.serverContext();
		const deviceId = await resolveMicrophone(
			String(config[SETTINGS_KEYS.VOICE_MICROPHONE] ?? VOICE_MICROPHONE_AUTO)
		);
		const constraints = request.constraints ?? VoiceService.processing(config);

		return {
			capture: {
				boost: config[SETTINGS_KEYS.VOICE_BOOST] !== false,
				...(deviceId ? { deviceId } : {}),
				...(constraints ? { constraints } : {})
			},
			realtime,
			transcribe:
				request.transcribe === false
					? undefined
					: {
							context,
							language: VoiceService.language(config),
							model: request.isRouter && request.model ? request.model : undefined
						}
		};
	}
}
