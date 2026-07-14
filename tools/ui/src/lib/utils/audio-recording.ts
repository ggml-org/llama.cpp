import { MimeTypeAudio } from '$lib/enums';

/**
 * AudioRecorder - Browser-based audio recording with MediaRecorder API
 *
 * This class provides a complete audio recording solution using the browser's MediaRecorder API.
 * It handles microphone access, recording state management, and audio format optimization.
 *
 * **Features:**
 * - Automatic microphone permission handling
 * - Audio enhancement (echo cancellation, noise suppression, auto gain)
 * - Multiple format support with fallback (WAV, WebM, MP4, AAC)
 * - Real-time recording state tracking
 * - Proper cleanup and resource management
 */
export class AudioRecorder {
	private mediaRecorder: MediaRecorder | null = null;
	private audioChunks: Blob[] = [];
	private stream: MediaStream | null = null;
	private recordingState: boolean = false;

	async startRecording(): Promise<void> {
		try {
			this.stream = await navigator.mediaDevices.getUserMedia({
				audio: {
					echoCancellation: true,
					noiseSuppression: true,
					autoGainControl: true
				}
			});

			this.initializeRecorder(this.stream);

			this.audioChunks = [];
			// Start recording with a small timeslice to ensure we get data
			this.mediaRecorder!.start(100);
			this.recordingState = true;
		} catch (error) {
			console.error('Failed to start recording:', error);
			throw new Error('Failed to access microphone. Please check permissions.');
		}
	}

	async stopRecording(): Promise<Blob> {
		return new Promise((resolve, reject) => {
			const recorder = this.mediaRecorder;
			const chunks = this.audioChunks;
			const stream = this.stream;

			if (!recorder || recorder.state === 'inactive') {
				reject(new Error('No active recording to stop'));
				return;
			}

			// Detach instance state right away so a new startRecording can take over without race
			this.mediaRecorder = null;
			this.audioChunks = [];
			this.stream = null;
			this.recordingState = false;

			recorder.onstop = () => {
				const audioBlob = new Blob(chunks, {
					type: recorder.mimeType || MimeTypeAudio.WAV
				});

				if (stream) {
					for (const track of stream.getTracks()) {
						track.stop();
					}
				}

				resolve(audioBlob);
			};

			recorder.onerror = (event) => {
				console.error('Recording error:', event);

				if (stream) {
					for (const track of stream.getTracks()) {
						track.stop();
					}
				}

				reject(new Error('Recording failed'));
			};

			recorder.stop();
		});
	}

	isRecording(): boolean {
		return this.recordingState;
	}

	cancelRecording(): void {
		const recorder = this.mediaRecorder;
		const stream = this.stream;

		this.mediaRecorder = null;
		this.audioChunks = [];
		this.stream = null;
		this.recordingState = false;

		if (recorder && recorder.state !== 'inactive') {
			// Drop the original handlers so the pending stop event does not touch the instance
			recorder.onstop = null;
			recorder.onerror = null;
			recorder.stop();
		}

		if (stream) {
			for (const track of stream.getTracks()) {
				track.stop();
			}
		}
	}

	private initializeRecorder(stream: MediaStream): void {
		const options: MediaRecorderOptions = {};

		if (MediaRecorder.isTypeSupported(MimeTypeAudio.WAV)) {
			options.mimeType = MimeTypeAudio.WAV;
		} else if (MediaRecorder.isTypeSupported(MimeTypeAudio.WEBM_OPUS)) {
			options.mimeType = MimeTypeAudio.WEBM_OPUS;
		} else if (MediaRecorder.isTypeSupported(MimeTypeAudio.WEBM)) {
			options.mimeType = MimeTypeAudio.WEBM;
		} else if (MediaRecorder.isTypeSupported(MimeTypeAudio.MP4)) {
			options.mimeType = MimeTypeAudio.MP4;
		} else {
			console.warn('No preferred audio format supported, using default');
		}

		this.mediaRecorder = new MediaRecorder(stream, options);

		this.mediaRecorder.ondataavailable = (event) => {
			if (event.data.size > 0) {
				this.audioChunks.push(event.data);
			}
		};

		this.mediaRecorder.onstop = () => {
			this.recordingState = false;
		};

		this.mediaRecorder.onerror = (event) => {
			console.error('MediaRecorder error:', event);
			this.recordingState = false;
		};
	}
}

export async function convertToWav(audioBlob: Blob): Promise<Blob> {
	try {
		if (audioBlob.type.includes('wav')) {
			return audioBlob;
		}

		const arrayBuffer = await audioBlob.arrayBuffer();

		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();

		try {
			const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
			return audioBufferToWav(audioBuffer);
		} finally {
			audioContext.close();
		}
	} catch (error) {
		console.error('Failed to convert audio to WAV:', error);
		return audioBlob;
	}
}

function audioBufferToWav(buffer: AudioBuffer): Blob {
	const length = buffer.length;
	const numberOfChannels = buffer.numberOfChannels;
	const sampleRate = buffer.sampleRate;
	const bytesPerSample = 2; // 16-bit
	const blockAlign = numberOfChannels * bytesPerSample;
	const byteRate = sampleRate * blockAlign;
	const dataSize = length * blockAlign;
	const bufferSize = 44 + dataSize;

	const arrayBuffer = new ArrayBuffer(bufferSize);
	const view = new DataView(arrayBuffer);

	const writeString = (offset: number, string: string) => {
		for (let i = 0; i < string.length; i++) {
			view.setUint8(offset + i, string.charCodeAt(i));
		}
	};

	writeString(0, 'RIFF'); // ChunkID
	view.setUint32(4, bufferSize - 8, true); // ChunkSize
	writeString(8, 'WAVE'); // Format
	writeString(12, 'fmt '); // Subchunk1ID
	view.setUint32(16, 16, true); // Subchunk1Size
	view.setUint16(20, 1, true); // AudioFormat (PCM)
	view.setUint16(22, numberOfChannels, true); // NumChannels
	view.setUint32(24, sampleRate, true); // SampleRate
	view.setUint32(28, byteRate, true); // ByteRate
	view.setUint16(32, blockAlign, true); // BlockAlign
	view.setUint16(34, 16, true); // BitsPerSample
	writeString(36, 'data'); // Subchunk2ID
	view.setUint32(40, dataSize, true); // Subchunk2Size

	// Cache channel arrays, write PCM via Int16Array (native little-endian, matches WAV)
	const channels: Float32Array[] = new Array(numberOfChannels);
	for (let c = 0; c < numberOfChannels; c++) {
		channels[c] = buffer.getChannelData(c);
	}

	const pcm = new Int16Array(arrayBuffer, 44, length * numberOfChannels);
	let p = 0;
	for (let i = 0; i < length; i++) {
		for (let c = 0; c < numberOfChannels; c++) {
			let s = channels[c][i];
			if (s > 1) s = 1;
			else if (s < -1) s = -1;
			pcm[p++] = s * 0x7fff;
		}
	}

	return new Blob([arrayBuffer], { type: MimeTypeAudio.WAV });
}

/**
 * Create a File object from audio blob with timestamp-based naming
 * @param audioBlob - The audio blob to wrap
 * @param filename - Optional custom filename
 * @returns File object with appropriate name and metadata
 */
export function createAudioFile(audioBlob: Blob, filename?: string): File {
	const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
	const extension = audioBlob.type.includes('wav') ? 'wav' : 'mp3';
	const defaultFilename = `recording-${timestamp}.${extension}`;

	return new File([audioBlob], filename || defaultFilename, {
		type: audioBlob.type,
		lastModified: Date.now()
	});
}

/**
 * Check if audio recording is supported in the current browser
 * @returns True if MediaRecorder and getUserMedia are available
 */
export function isAudioRecordingSupported(): boolean {
	return !!(
		typeof navigator !== 'undefined' &&
		navigator.mediaDevices &&
		typeof navigator.mediaDevices.getUserMedia === 'function' &&
		typeof window !== 'undefined' &&
		window.MediaRecorder
	);
}
