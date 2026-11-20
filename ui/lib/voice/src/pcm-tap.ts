import {
	VOICE_ERROR_MESSAGE,
	VOICE_SCRIPT_PROCESSOR_FRAMES,
	VOICE_WORKLET_BATCH_FRAMES,
	VOICE_WORKLET_FLUSH_TIMEOUT_MS,
	VOICE_WORKLET_MESSAGE,
	VOICE_WORKLET_PROCESSOR,
	VOICE_WORKLET_SOURCE_MIME
} from './constants.js';

// runs on the audio thread; inline so no extra file has to be served next to the app
const WORKLET_SOURCE = `
class PcmTap extends AudioWorkletProcessor {
	constructor(options) {
		super();
		this.size = options.processorOptions.batchFrames;
		this.buffer = new Float32Array(this.size);
		this.filled = 0;
		this.port.onmessage = (event) => {
			if (event.data !== '${VOICE_WORKLET_MESSAGE.FLUSH}') return;
			this.post();
			this.port.postMessage('${VOICE_WORKLET_MESSAGE.FLUSHED}');
		};
	}

	post() {
		if (this.filled === 0) return;
		const block = this.buffer.slice(0, this.filled);
		this.port.postMessage(block, [block.buffer]);
		this.filled = 0;
	}

	process(inputs) {
		const channel = inputs[0] && inputs[0][0];
		if (!channel) return true;
		let offset = 0;
		while (offset < channel.length) {
			const n = Math.min(channel.length - offset, this.size - this.filled);
			this.buffer.set(channel.subarray(offset, offset + n), this.filled);
			this.filled += n;
			offset += n;
			if (this.filled === this.size) this.post();
		}
		return true;
	}
}
registerProcessor('${VOICE_WORKLET_PROCESSOR}', PcmTap);
`;

/** A node that hands mono float blocks of its input to the main thread. */
export interface PcmTap {
	node: AudioNode;
	/** Posts what is still buffered, then resolves */
	flush(): Promise<void>;
	disconnect(): void;
}

/**
 * Taps mono float samples from the graph: an AudioWorklet loaded from a Blob URL
 * (fine under COOP/COEP, a blob inherits the page's origin), or a ScriptProcessor
 * where AudioWorklet is missing.
 */
export async function createPcmTap(
	context: AudioContext,
	onBlock: (block: Float32Array) => void
): Promise<PcmTap> {
	if (context.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
		try {
			return await createWorkletTap(context, onBlock);
		} catch (error) {
			console.warn('AudioWorklet unavailable, using ScriptProcessor:', error);
		}
	}

	return createScriptProcessorTap(context, onBlock);
}

// a context loads the worklet once, however many recordings it serves
const modules = new WeakMap<BaseAudioContext, Promise<void>>();

/** Loads the tap's worklet into `context`, once; later calls wait for the same load. */
export function loadPcmTapModule(context: BaseAudioContext): Promise<void> {
	let loading = modules.get(context);

	if (!loading) {
		if (!context.audioWorklet)
			return Promise.reject(new Error(VOICE_ERROR_MESSAGE.workletUnavailable));

		const url = URL.createObjectURL(
			new Blob([WORKLET_SOURCE], { type: VOICE_WORKLET_SOURCE_MIME })
		);

		loading = context.audioWorklet.addModule(url).finally(() => URL.revokeObjectURL(url));
		// a failed load may be tried again
		loading.catch(() => modules.delete(context));
		modules.set(context, loading);
	}

	return loading;
}

async function createWorkletTap(
	context: AudioContext,
	onBlock: (block: Float32Array) => void
): Promise<PcmTap> {
	await loadPcmTapModule(context);

	const node = new AudioWorkletNode(context, VOICE_WORKLET_PROCESSOR, {
		channelCount: 1,
		channelCountMode: 'explicit',
		numberOfInputs: 1,
		numberOfOutputs: 1,
		processorOptions: { batchFrames: VOICE_WORKLET_BATCH_FRAMES }
	});

	let flushed: (() => void) | null = null;

	node.port.onmessage = (event: MessageEvent<Float32Array | string>) => {
		if (event.data === VOICE_WORKLET_MESSAGE.FLUSHED) {
			flushed?.();
		} else if (event.data instanceof Float32Array) {
			onBlock(event.data);
		}
	};

	return {
		disconnect: () => {
			node.port.onmessage = null;
			node.port.close();
			node.disconnect();
		},
		flush: () =>
			new Promise<void>((resolve) => {
				const timer = setTimeout(resolve, VOICE_WORKLET_FLUSH_TIMEOUT_MS);

				flushed = () => {
					clearTimeout(timer);
					flushed = null;
					resolve();
				};
				node.port.postMessage(VOICE_WORKLET_MESSAGE.FLUSH);
			}),
		node
	};
}

function createScriptProcessorTap(
	context: AudioContext,
	onBlock: (block: Float32Array) => void
): PcmTap {
	const node = context.createScriptProcessor(VOICE_SCRIPT_PROCESSOR_FRAMES, 1, 1);

	node.onaudioprocess = (event) => {
		onBlock(event.inputBuffer.getChannelData(0).slice());
	};

	return {
		disconnect: () => {
			node.onaudioprocess = null;
			node.disconnect();
		},
		// a ScriptProcessor holds nothing back: each block is posted as soon as it is full
		flush: () => Promise.resolve(),
		node
	};
}
