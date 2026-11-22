import type { VoiceSessionOptions, VoiceSessionResult } from '@ggml-org/llama-voice';
import { VoiceRecorderStatus } from '@ggml-org/llama-voice';
import type { VoiceRecorderLike } from '#lib/hooks/use-chat-form-voice.svelte.ts';

/** A transcript the test resolves when it wants. */
export function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (error: unknown) => void;

	const promise = new Promise<T>((res, rej) => {
		resolve = res;
		reject = rej;
	});

	return { promise, reject, resolve };
}

/** A silent 16-bit mono WAV of `ms`, a valid clip to attach and play. */
export function silentWav(ms = 500, sampleRate = 8000): Blob {
	const samples = Math.round((ms / 1000) * sampleRate);
	const buffer = new ArrayBuffer(44 + samples * 2);
	const view = new DataView(buffer);
	const write = (offset: number, text: string) => {
		for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
	};

	write(0, 'RIFF');
	view.setUint32(4, 36 + samples * 2, true);
	write(8, 'WAVE');
	write(12, 'fmt ');
	view.setUint32(16, 16, true);
	view.setUint16(20, 1, true);
	view.setUint16(22, 1, true);
	view.setUint32(24, sampleRate, true);
	view.setUint32(28, sampleRate * 2, true);
	view.setUint16(32, 2, true);
	view.setUint16(34, 16, true);
	write(36, 'data');
	view.setUint32(40, samples * 2, true);

	return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * A VoiceRecorder without a microphone: the test drives its levels and live text and
 * decides the transcript, and reads the options the form started it with.
 */
export class FakeVoiceRecorder implements VoiceRecorderLike {
	cancelled = 0;
	levels = $state.raw<number[]>(new Array(48).fill(0));
	live = $state(false);
	liveText = $state('');
	options: VoiceSessionOptions | undefined;

	quiet = $state(false);
	startError: Error | null = null;
	status = $state<VoiceRecorderStatus>(VoiceRecorderStatus.IDLE);
	transcript: Promise<string> | null = null;

	cancel(): void {
		this.cancelled++;
		this.status = VoiceRecorderStatus.IDLE;
	}

	/** Live text as a realtime socket would stream it. */
	say(text: string): void {
		this.live = true;
		this.liveText = text;
		this.options?.onText?.(text);
	}

	setLevels(levels: number[]): void {
		this.levels = levels;
		this.options?.onLevels?.(levels);
	}

	/** The speech turning too quiet, or loud enough again, as the capture would tell. */
	setQuiet(quiet: boolean): void {
		this.quiet = quiet;
		this.options?.onQuiet?.(quiet);
	}

	async start(options: VoiceSessionOptions = {}): Promise<void> {
		if (this.startError) throw this.startError;

		this.options = options;
		this.liveText = '';
		this.status = VoiceRecorderStatus.RECORDING;
	}

	async stop(): Promise<VoiceSessionResult> {
		this.status = VoiceRecorderStatus.IDLE;

		return {
			clip: {
				blob: silentWav(),
				durationMs: 2400,
				peaks: Array.from({ length: 64 }, (_, i) => Math.abs(Math.sin(i / 4))),
				sampleRate: 8000
			},
			liveText: this.liveText.trim() || null,
			transcript: this.options?.transcribe ? this.transcript : null
		};
	}
}

/**
 * A fake recorder for stories: once started it moves its levels like someone speaking
 * and, when given a script, streams it word by word as a realtime model would.
 */
export class DemoVoiceRecorder extends FakeVoiceRecorder {
	private timer: ReturnType<typeof setInterval> | undefined;

	override cancel(): void {
		clearInterval(this.timer);
		super.cancel();
	}

	constructor(
		private script = '',
		private level: (t: number) => number = () => Math.random()
	) {
		super();
	}

	override async start(options: VoiceSessionOptions = {}): Promise<void> {
		await super.start(options);

		const words = this.script.split(' ').filter(Boolean);
		const startedAt = Date.now();

		let history = new Array(48).fill(0);

		this.timer = setInterval(() => {
			const t = (Date.now() - startedAt) / 1000;
			const spoken = Math.min(Math.floor(t * 2.5), words.length);

			history = [...history.slice(1), this.level(t)];
			this.setLevels(history);

			if (spoken > 0 && words.slice(0, spoken).join(' ') !== this.liveText) {
				this.say(words.slice(0, spoken).join(' '));
			}
		}, 50);
	}

	override async stop(): Promise<VoiceSessionResult> {
		clearInterval(this.timer);

		if (!this.transcript && this.script) this.transcript = Promise.resolve(this.script);

		return super.stop();
	}
}
