import {
	VOICE_ANALYSER_FFT_SIZE,
	VOICE_CHUNK_MS,
	VOICE_LEVELS_INTERVAL_MS,
	VOICE_LEVELS_LENGTH,
	VOICE_MICROPHONE_CONSTRAINTS,
	VOICE_PEAKS_COUNT,
	VOICE_SAMPLE_RATE
} from './constants';
import { VoiceCaptureError } from './errors';
import { LevelHistory, rms, rmsToLevel } from './levels';
import { concatPcm16, createResampler, floatToPcm16, type Resampler } from './pcm';
import { createPcmTap, type PcmTap } from './pcm-tap';
import { computePeaks } from './peaks';
import type {
	VoiceCaptureOptions,
	VoiceCaptureState,
	VoiceClip,
	VoiceLevelsListener,
	VoicePcmListener
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
 * returned by `stop()` as a WAV clip with its peaks. Every path (stop, cancel, a
 * failed start, the page going away) releases the tracks, the nodes and the context.
 */
export class VoiceCapture {
	readonly sampleRate: number;

	private chunks: Int16Array[] = [];
	private readonly chunkSamples: number;
	private currentState: VoiceCaptureState = 'idle';
	private frame: number | null = null;
	// bumped by every start, stop and cancel, so a start still awaiting the microphone can tell it was dropped
	private generation = 0;

	private graph: CaptureGraph | null = null;
	private handlePageHide = () => this.cancel();
	private readonly history: LevelHistory;
	private levelListeners = new Set<VoiceLevelsListener>();
	private readonly levelsIntervalMs: number;
	private readonly options: VoiceCaptureOptions;
	private partial: number[] = [];
	private pcmListeners = new Set<VoicePcmListener>();
	private readonly peaksCount: number;
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

	/** Drops the capture, including one still asking for the microphone. */
	cancel(): void {
		this.generation++;
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

		let stream: MediaStream | null = this.options.stream ?? null;
		let context: AudioContext | null = null;
		let started = false;

		try {
			stream ??= await navigator.mediaDevices.getUserMedia({
				audio: { ...VOICE_MICROPHONE_CONSTRAINTS, ...this.options.constraints }
			});

			if (dropped()) throw new DOMException('Voice capture cancelled', 'AbortError');

			context = new AudioContext();

			// a context made outside a user gesture starts suspended; resume() resolves once allowed
			if (context.state === 'suspended') context.resume().catch(() => {});

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

		const samples = concatPcm16(this.chunks);

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

		const chunk = floatToPcm16(Float32Array.from(this.partial));

		this.partial = [];
		this.chunks.push(chunk);
		this.recordedSamples += chunk.length;

		for (const listener of this.pcmListeners) listener(chunk);
	}

	private handleBlock(block: Float32Array): void {
		if (!this.resampler || this.currentState !== 'recording') return;

		const resampled = this.resampler.process(block);

		for (let i = 0; i < resampled.length; i++) {
			this.partial.push(resampled[i]);

			if (this.partial.length === this.chunkSamples) this.flushPartial();
		}
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

	if (context && context.state !== 'closed') {
		context.close().catch(() => {});
	}
}
