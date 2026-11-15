import { VoiceCapture } from './capture';
import { RealtimeUnsupportedError } from './errors';
import { RealtimeTranscriber } from './realtime';
import { transcribeClip } from './transcribe';
import type { VoiceSession, VoiceSessionOptions, VoiceSessionResult } from './types';

/**
 * Starts capturing and, when asked to transcribe, tries the realtime socket at the
 * same time. Realtime gives live text; when the model cannot stream, the socket does
 * not open, or the session fails, the clip is transcribed over HTTP after recording.
 * @throws VoiceCaptureError when the microphone does not start
 */
export async function startVoiceSession(options: VoiceSessionOptions = {}): Promise<VoiceSession> {
	const capture = isCapture(options.capture) ? options.capture : new VoiceCapture(options.capture);
	const unsubscribers: (() => void)[] = [];

	if (options.onLevels) unsubscribers.push(capture.onLevels(options.onLevels));

	try {
		await capture.start();
	} catch (error) {
		for (const unsubscribe of unsubscribers) unsubscribe();

		throw error;
	}

	const transcribe = options.transcribe;

	let transcriber: RealtimeTranscriber | null = null;
	let liveText: string | null = null;
	let ended = false;

	if (transcribe && options.realtime !== false) {
		const realtime = new RealtimeTranscriber(transcribe.context, {
			autoload: transcribe.autoload,
			language: transcribe.language,
			model: transcribe.model,
			onCompleted: (text) => {
				liveText = text;
				options.onText?.(text);
			},
			onDelta: (_delta, text) => {
				liveText = text;
				options.onText?.(text);
			},
			sampleRate: capture.sampleRate
		});
		const unsubscribePcm = capture.onPcm((chunk) => realtime.append(chunk));

		transcriber = realtime;
		unsubscribers.push(unsubscribePcm);
		realtime.connect().then(
			() => {
				if (!ended) options.onRealtime?.(realtime.live);
			},
			(error) => {
				unsubscribePcm();

				if (ended) return;

				if (!(error instanceof RealtimeUnsupportedError)) {
					console.warn('Realtime transcription unavailable, transcribing after recording:', error);
				}

				options.onRealtime?.(false);
			}
		);
	} else if (transcribe) {
		options.onRealtime?.(false);
	}

	const release = () => {
		ended = true;

		for (const unsubscribe of unsubscribers) unsubscribe();

		unsubscribers.length = 0;
	};

	return {
		cancel() {
			release();
			capture.cancel();
			transcriber?.close();
		},
		capture,
		async stop(): Promise<VoiceSessionResult> {
			let clip;

			try {
				// the last PCM chunk reaches the socket before stop() resolves
				clip = await capture.stop();
			} catch (error) {
				release();
				transcriber?.close();

				throw error;
			}

			release();

			if (!transcribe) return { clip, liveText: null, transcript: null };

			const overHttp = () =>
				transcribeClip(transcribe.context, clip.blob, {
					language: transcribe.language,
					model: transcribe.model
				});
			const realtime = transcriber;

			let transcript: Promise<string>;

			if (realtime?.ready) {
				transcript = realtime.finish().catch((error) => {
					console.warn('Realtime transcript failed, transcribing over HTTP:', error);

					return overHttp();
				});
			} else {
				// still connecting at the end: the HTTP request is the shorter wait
				realtime?.close();
				transcript = overHttp();
			}

			return { clip, liveText: liveText?.trim() || null, transcript };
		}
	};
}

function isCapture(value: VoiceSessionOptions['capture']): value is VoiceCapture {
	return typeof (value as VoiceCapture | undefined)?.start === 'function';
}
