import { VoiceCapture } from './capture.js';
import { VOICE_CLIP_FILENAME, VOICE_PIECE, VOICE_RETRANSCRIBE_MAX_MS } from './constants.js';
import { RealtimeUnsupportedError } from './errors.js';
import { HttpPieceTranscriber } from './http-pieces.js';
import { PauseDetector } from './pauses.js';
import { RealtimeTranscriber } from './realtime.js';
import { transcribeClip } from './transcribe.js';
import type { VoiceSession, VoiceSessionOptions, VoiceSessionResult } from './types.js';

/**
 * Starts capturing and, when asked to transcribe, tries the realtime socket at the
 * same time. Realtime gives live text: a live ASR model streams it, any other model
 * that hears transcribes the recording piece by piece, cut at the speaker's pauses.
 * When the model cannot hold a realtime session (Qwen3-ASR) or the socket does not
 * open, the pieces go over HTTP instead, one request each, so the text is live too.
 * Either way, a recording cut into pieces gets one more pass over the whole clip
 * at the end, punctuated by meaning rather than by the pauses.
 * @throws VoiceCaptureError when the microphone does not start
 */
export async function startVoiceSession(options: VoiceSessionOptions = {}): Promise<VoiceSession> {
	const capture = isCapture(options.capture) ? options.capture : new VoiceCapture(options.capture);
	const unsubscribers: (() => void)[] = [];

	if (options.onLevels) unsubscribers.push(capture.onLevels(options.onLevels));

	if (options.onQuiet) unsubscribers.push(capture.onQuiet(options.onQuiet));

	try {
		await capture.start();
	} catch (error) {
		for (const unsubscribe of unsubscribers) unsubscribe();

		throw error;
	}

	const transcribe = options.transcribe;

	let transcriber: RealtimeTranscriber | null = null;
	// speech heard since the last piece was committed; until pieces are cut, assume so
	let speechSinceCommit = true;
	let liveText: string | null = null;
	let ended = false;

	const showText = (text: string) => {
		liveText = text;
		options.onText?.(text);
	};
	// cuts the recording at the speaker's pauses: short ones, so the text keeps up with them
	const cutAtPauses = (cut: () => boolean) => {
		const pauses = new PauseDetector({
			intervalMs: capture.levelsIntervalMs,
			maxSegmentMs: VOICE_PIECE.MAX_MS,
			minSpeechMs: VOICE_PIECE.MIN_SPEECH_MS,
			pauseMs: VOICE_PIECE.PAUSE_MS
		});

		unsubscribers.push(
			capture.onLevels((levels) => {
				const atPause = pauses.push(levels[levels.length - 1] ?? 0);

				if (pauses.heardSpeech) speechSinceCommit = true;

				if (atPause && cut()) speechSinceCommit = false;
			})
		);
	};
	// pieces over HTTP, for when realtime is not to be had; the audio is kept from the start,
	// so nothing said while the socket answers is lost if the model turns out unable to stream
	const httpPieces = transcribe
		? new HttpPieceTranscriber({
				onText: showText,
				sampleRate: capture.sampleRate,
				transcribe: (wav, signal) =>
					transcribeClip(transcribe.context, wav, {
						filename: VOICE_CLIP_FILENAME.PIECE,
						language: transcribe.language,
						model: transcribe.model,
						signal
					})
			})
		: null;

	let piecesOverHttp = false;

	if (httpPieces) unsubscribers.push(capture.onPcm((chunk) => httpPieces.append(chunk)));

	const startHttpPieces = () => {
		if (!httpPieces || ended) return;

		piecesOverHttp = true;
		cutAtPauses(() => {
			if (!httpPieces.canCut) return false;

			httpPieces.cut();

			return true;
		});
	};

	if (transcribe && options.realtime !== false) {
		const realtime = new RealtimeTranscriber(transcribe.context, {
			autoload: transcribe.autoload,
			language: transcribe.language,
			model: transcribe.model,
			onCompleted: showText,
			onDelta: (_delta, text) => showText(text),
			sampleRate: capture.sampleRate
		});
		const unsubscribePcm = capture.onPcm((chunk) => realtime.append(chunk));

		transcriber = realtime;
		unsubscribers.push(unsubscribePcm);
		realtime.connect().then(
			() => {
				// the socket has it all from here
				httpPieces?.close();

				if (ended) return;

				options.onRealtime?.(realtime.live);

				// a model that transcribes per commit gets the speech piece by piece, at its pauses
				if (!realtime.live) {
					cutAtPauses(() => {
						if (!realtime.canCommitSegment) return false;

						realtime.commitSegment();

						return true;
					});
				}
			},
			(error) => {
				unsubscribePcm();

				if (ended) return;

				if (!(error instanceof RealtimeUnsupportedError)) {
					console.warn('Realtime transcription unavailable, transcribing over HTTP:', error);
				}

				options.onRealtime?.(false);
				startHttpPieces();
			}
		);
	} else if (transcribe) {
		options.onRealtime?.(false);
		startHttpPieces();
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
			httpPieces?.close();
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
				httpPieces?.close();

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
				// more than one piece: the text was cut at pauses, maybe inside a sentence
				const split = !realtime.live && realtime.piecesCommitted + (speechSinceCommit ? 1 : 0) > 1;
				// a silent rest is left out: a model may hear a word ("yeah") in silence
				const pieces = realtime.finish({ commitRest: speechSinceCommit }).catch((error) => {
					console.warn('Realtime transcript failed, transcribing over HTTP:', error);

					return overHttp();
				});

				transcript =
					split && clip.durationMs <= VOICE_RETRANSCRIBE_MAX_MS
						? // the whole clip once more, punctuated by meaning, not by the pauses
							overHttp().then(
								(text) => text || pieces,
								(error) => {
									console.warn(
										'Transcribing the whole recording failed, keeping its pieces:',
										error
									);

									return pieces;
								}
							)
						: pieces;
			} else if (piecesOverHttp && httpPieces) {
				// the whole clip now, punctuated by meaning; a piece still on its way would hold the server up
				const fromPieces = httpPieces.text;

				httpPieces.close();
				transcript = overHttp().catch((error) => {
					if (!fromPieces) throw error;

					console.warn('Transcribing the whole recording failed, keeping its pieces:', error);

					return fromPieces;
				});
			} else {
				// still connecting at the end: the HTTP request is the shorter wait
				realtime?.close();
				httpPieces?.close();
				transcript = overHttp();
			}

			return { clip, liveText: liveText?.trim() || null, transcript };
		}
	};
}

function isCapture(value: VoiceSessionOptions['capture']): value is VoiceCapture {
	return typeof (value as VoiceCapture | undefined)?.start === 'function';
}
