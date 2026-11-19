import { captureContext, idleCaptureContext } from './audio-context';
import {
	VOICE_ANALYSER_FFT_SIZE,
	VOICE_CHUNK_MS,
	VOICE_FIRST_AUDIO_TIMEOUT_MS,
	VOICE_LEVELS_INTERVAL_MS,
	VOICE_LEVELS_LENGTH,
	VOICE_MICROPHONE_CONSTRAINTS,
	VOICE_PEAKS_COUNT,
	VOICE_SAMPLE_RATE
} from './constants';
import { VoiceCaptureError } from './errors';
import { normalizeSpeech, SpeechGain } from './gain';
import { LevelHistory, rms, rmsToLevel } from './levels';
import { concatPcm16, createResampler, floatToPcm16, type Resampler } from './pcm';
import { createPcmTap, type PcmTap } from './pcm-tap';
import { computePeaks } from './peaks';
import type {
	VoiceCaptureOptions,
	VoiceCaptureState,
	VoiceClip,
	VoiceLevelsListener,
	VoicePcmListener,
	VoiceQuietListener
} from './types';
import { encodeWav } from './wav';
import { describeMicrophoneError } from '#lib/utils/audio-recording.ts';

/** Graph and device handles of a running capture, released together. */
interface CaptureGraph {
	analyser: AnalyserNode;
	context: AudioContext;
	mute: GainNode;
	source: MediaStreamAudioSourceNode;
	stream: MediaStream;
	tap: PcmTap;
}

/**
 * Microphone capture for voice input. One `start()` gives, at once: a live level
 * history for the waveform, PCM16 mono chunks for streaming, and the recording,
 * returned by `stop()` as a WAV clip with its peaks. With `boost`, quiet speech is
 * raised (gain.ts): the chunks as they stream, the clip in one pass when it stops.
 * Whether the speech is too quiet even so is told either way (onQuiet). `start()` resolves once audio
 * flows, so what is said after it is in the clip. Every path (stop, cancel, a failed
 * start, the page going away) releases the tracks and the nodes; the page's shared
 * AudioContext is suspended for the next recording (see audio-context.ts).
 */
export class VoiceCapture {
	/** Time between two level values. */
	readonly levelsIntervalMs: number;

	readonly sampleRate: number;
	private chunks: Int16Array[] = [];
	private readonly chunkSamples: number;
	private currentState: VoiceCaptureState = 'idle';
	// resolves the wait of start() for the first audio
	private firstAudio: (() => void) | null = null;
	private frame: number | null = null;
	private gain: SpeechGain | null = null;

	// bumped by every start, stop and cancel, so a start still awaiting the microphone can tell it was dropped
	private generation = 0;
	private graph: CaptureGraph | null = null;
	private handlePageHide = () => this.cancel();
	private readonly history: LevelHistory;
	private levelListeners = new Set<VoiceLevelsListener>();
	private readonly options: VoiceCaptureOptions;
	private partial: number[] = [];
	private pcmListeners = new Set<VoicePcmListener>();
	private readonly peaksCount: number;
	private quiet = false;
	private quietListeners = new Set<VoiceQuietListener>();
	private recordedSamples = 0;

	private resampler: Resampler | null = null;

	/** Length of the audio captured so far. */
	get durationMs(): number {
		return ((this.recordedSamples + this.partial.length) / this.sampleRate) * 1000;
	}

	/** The level history now, 0..1, newest last. */
	get levels(): number[] {
		return this.history.snapshot();
	}

	get state(): VoiceCaptureState {
		return this.currentState;
	}

	/** Whether the speech heard so far stays quiet even at the highest gain. */
	get tooQuiet(): boolean {
		return this.quiet;
	}

	/** Drops the capture, including one still asking for the microphone. */
	cancel(): void {
		this.generation++;
		this.firstAudio?.();
		this.teardown();
		this.chunks = [];
		this.partial = [];
		this.recordedSamples = 0;
		this.currentState = 'idle';
	}

	constructor(options: VoiceCaptureOptions = {}) {
		this.options = options;
		this.sampleRate = options.sampleRate ?? VOICE_SAMPLE_RATE;
		this.chunkSamples = Math.max(
			1,
			Math.round((this.sampleRate * (options.chunkMs ?? VOICE_CHUNK_MS)) / 1000)
		);
		this.history = new LevelHistory(options.levelsLength ?? VOICE_LEVELS_LENGTH);
		this.levelsIntervalMs = options.levelsIntervalMs ?? VOICE_LEVELS_INTERVAL_MS;
		this.peaksCount = options.peaksCount ?? VOICE_PEAKS_COUNT;
	}

	/** Listens to the level history; returns the unsubscribe function. */
	onLevels(listener: VoiceLevelsListener): () => void {
		this.levelListeners.add(listener);

		return () => this.levelListeners.delete(listener);
	}

	/** Listens to the PCM16 chunks; returns the unsubscribe function. */
	onPcm(listener: VoicePcmListener): () => void {
		this.pcmListeners.add(listener);

		return () => this.pcmListeners.delete(listener);
	}

	/** Listens to whether the speech is too quiet, told when it changes; returns the unsubscribe function. */
	onQuiet(listener: VoiceQuietListener): () => void {
		this.quietListeners.add(listener);

		return () => this.quietListeners.delete(listener);
	}

	/**
	 * Asks for the microphone and starts capturing.
	 * @throws VoiceCaptureError with a message for the user when the microphone does not start
	 */
	async start(): Promise<void> {
		if (this.currentState === 'starting' || this.currentState === 'recording') {
			throw new Error('Voice capture already started');
		}

		const generation = ++this.generation;
		const dropped = () => generation !== this.generation;

		this.currentState = 'starting';
		this.chunks = [];
		this.partial = [];
		this.recordedSamples = 0;
		this.history.reset();
		this.gain = new SpeechGain();
		this.setQuiet(false);

		let stream: MediaStream | null = this.options.stream ?? null;
		let context: AudioContext | null = null;
		let started = false;

		const timing = startTiming();

		try {
			const shared = captureContext();

			context = shared.context;

			// resumed now, while the click still allows it: the microphone may take a while
			if (context.state === 'suspended') context.resume().catch(() => {});

			const deviceId = this.options.deviceId;

			stream ??= await navigator.mediaDevices.getUserMedia({
				audio: {
					...VOICE_MICROPHONE_CONSTRAINTS,
					...(deviceId ? { deviceId } : {}),
					...this.options.constraints
				}
			});
			timing.mark('microphone');
			timing.note(
				checkProcessing(stream, { ...VOICE_MICROPHONE_CONSTRAINTS, ...this.options.constraints })
			);

			if (dropped()) throw new DOMException('Voice capture cancelled', 'AbortError');

			await shared.ready;
			timing.mark('audio context');

			const source = context.createMediaStreamSource(stream);
			const analyser = context.createAnalyser();
			const mute = context.createGain();

			analyser.fftSize = VOICE_ANALYSER_FFT_SIZE;
			mute.gain.value = 0;
			this.resampler = createResampler(context.sampleRate, this.sampleRate);

			const tap = await createPcmTap(context, (block) => this.handleBlock(block));

			if (dropped()) {
				tap.disconnect();

				throw new DOMException('Voice capture cancelled', 'AbortError');
			}

			// the tap only runs while it is pulled by the destination, muted so nothing is heard
			source.connect(analyser);
			source.connect(tap.node);
			tap.node.connect(mute);
			mute.connect(context.destination);

			this.graph = { analyser, context, mute, source, stream, tap };
			started = true;
			this.currentState = 'recording';
			this.watchLevels(analyser);
			addEventListener('pagehide', this.handlePageHide);

			await this.waitForFirstAudio();
			timing.mark('first audio');
			timing.report();

			if (dropped()) throw new DOMException('Voice capture cancelled', 'AbortError');
		} catch (error) {
			if (!started) releaseDevice(stream, context);

			if (!dropped()) {
				this.currentState = 'idle';
			}

			if (error instanceof DOMException && error.name === 'AbortError' && dropped()) throw error;

			throw new VoiceCaptureError(describeMicrophoneError(error), { cause: error });
		}
	}

	/** Ends the capture and returns the recording. */
	async stop(): Promise<VoiceClip> {
		const graph = this.graph;

		if (!graph || this.currentState !== 'recording') {
			throw new Error('No voice capture to stop');
		}

		this.generation++;
		await graph.tap.flush();

		if (this.graph !== graph) throw new Error('Voice capture cancelled');

		this.flushPartial();
		this.teardown();

		const recorded = concatPcm16(this.chunks);
		// the whole recording at once: its first words get the gain the later ones set
		const samples = this.options.boost
			? normalizeSpeech(recorded, this.sampleRate).samples
			: recorded;

		this.chunks = [];
		this.currentState = 'stopped';

		return {
			blob: encodeWav(samples, this.sampleRate),
			durationMs: (samples.length / this.sampleRate) * 1000,
			peaks: computePeaks(samples, this.peaksCount),
			sampleRate: this.sampleRate
		};
	}

	private flushPartial(): void {
		if (this.partial.length === 0) return;

		const samples = Float32Array.from(this.partial);
		// the recording keeps the raw signal: stop() raises it as a whole
		const chunk = floatToPcm16(samples);
		const boost = this.options.boost === true;
		const streamed = this.gain?.process(samples, boost) ?? samples;

		this.partial = [];
		this.chunks.push(chunk);
		this.recordedSamples += chunk.length;
		this.setQuiet(this.gain?.tooQuiet ?? false);

		const sent = boost && streamed !== samples ? floatToPcm16(streamed) : chunk;

		for (const listener of this.pcmListeners) listener(sent);
	}

	private handleBlock(block: Float32Array): void {
		if (!this.resampler || this.currentState !== 'recording') return;

		this.firstAudio?.();

		const resampled = this.resampler.process(block);

		for (let i = 0; i < resampled.length; i++) {
			this.partial.push(resampled[i]);

			if (this.partial.length === this.chunkSamples) this.flushPartial();
		}
	}

	private setQuiet(quiet: boolean): void {
		if (quiet === this.quiet) return;

		this.quiet = quiet;

		for (const listener of this.quietListeners) listener(quiet);
	}

	private teardown(): void {
		removeEventListener('pagehide', this.handlePageHide);

		if (this.frame !== null) {
			cancelAnimationFrame(this.frame);
			this.frame = null;
		}

		const graph = this.graph;

		this.graph = null;
		this.resampler = null;

		if (!graph) return;

		graph.tap.disconnect();
		graph.source.disconnect();
		graph.analyser.disconnect();
		graph.mute.disconnect();
		releaseDevice(graph.stream, graph.context);
	}

	/** Resolves with the first audio block, or after a timeout so a silent device still records. */
	private waitForFirstAudio(): Promise<void> {
		return new Promise((resolve) => {
			const done = () => {
				clearTimeout(timer);
				this.firstAudio = null;
				resolve();
			};
			const timer = setTimeout(done, VOICE_FIRST_AUDIO_TIMEOUT_MS);

			this.firstAudio = done;
		});
	}

	private watchLevels(analyser: AnalyserNode): void {
		const buffer = new Float32Array(analyser.fftSize);

		let loudest = 0;
		let last = performance.now();

		const tick = (now: number) => {
			if (!this.graph) return;

			analyser.getFloatTimeDomainData(buffer);
			loudest = Math.max(loudest, rmsToLevel(rms(buffer)));

			if (now - last >= this.levelsIntervalMs) {
				const levels = this.history.push(loudest);

				loudest = 0;
				last = now;

				for (const listener of this.levelListeners) listener(levels);
			}

			this.frame = requestAnimationFrame(tick);
		};

		this.frame = requestAnimationFrame(tick);
	}
}

function releaseDevice(stream: MediaStream | null, context: AudioContext | null): void {
	for (const track of stream?.getTracks() ?? []) track.stop();

	if (context) idleCaptureContext(context);
}

const PROCESSING = ['echoCancellation', 'noiseSuppression', 'autoGainControl'] as const;

/**
 * What the browser gave for the microphone, e.g. "MacBook Pro Microphone, 48000 Hz, raw".
 * Processing that was asked off but is on is warned about: the browser keeps one setting
 * per device, so a page or app that has the same microphone open with echo cancellation
 * (a call, voice mode) gets the recording processed too, and the system may lower other
 * sound while it lasts.
 */
function checkProcessing(stream: MediaStream, asked: MediaTrackConstraints): string {
	const track = stream.getAudioTracks()[0];

	if (!track) return 'no audio track';

	const settings = track.getSettings();
	const on = PROCESSING.filter((key) => settings[key] === true);
	const unwanted = on.filter((key) => asked[key] === false);

	if (unwanted.length > 0) {
		console.warn(
			`[voice] ${track.label || 'The microphone'} is processed (${unwanted.join(', ')}) although the raw signal was asked for; another tab or app may have it open for a call.`
		);
	}

	const rate = settings.sampleRate ? `${settings.sampleRate} Hz` : null;

	return [track.label || 'microphone', rate, on.length > 0 ? on.join(' + ') : 'raw']
		.filter(Boolean)
		.join(', ');
}

/** How long each step of a start took, logged in development to see where a slow start goes. */
function startTiming() {
	const startedAt = performance.now();
	const steps: string[] = [];
	const notes: string[] = [];

	let last = startedAt;

	return {
		mark(step: string) {
			const now = performance.now();

			steps.push(`${step} ${Math.round(now - last)} ms`);
			last = now;
		},
		note(text: string) {
			notes.push(text);
		},
		report() {
			if (!import.meta.env?.DEV) return;

			console.debug(
				`[voice] recording after ${Math.round(performance.now() - startedAt)} ms: ${steps.join(', ')}${notes.length > 0 ? ` (${notes.join('; ')})` : ''}`
			);
		}
	};
}
