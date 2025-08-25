import { AudioMimeType } from "$lib/constants/supported-file-types";

export interface AudioRecordingOptions {
	mimeType?: string;
	audioBitsPerSecond?: number;
}

export class AudioRecorder {
	private mediaRecorder: MediaRecorder | null = null;
	private audioChunks: Blob[] = [];
	private stream: MediaStream | null = null;
	private recordingState: boolean = false;

	async startRecording(options: AudioRecordingOptions = {}): Promise<void> {
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
			if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
				reject(new Error('No active recording to stop'));
				return;
			}

			this.mediaRecorder.onstop = () => {
				const mimeType = this.mediaRecorder?.mimeType || AudioMimeType.WAV;
				const audioBlob = new Blob(this.audioChunks, { type: mimeType });
				
				this.cleanup();
				
				resolve(audioBlob);
			};

			this.mediaRecorder.onerror = (event) => {
				console.error('Recording error:', event);
				this.cleanup();
				reject(new Error('Recording failed'));
			};

			this.mediaRecorder.stop();
		});
	}

	isRecording(): boolean {
		return this.recordingState;
	}

	cancelRecording(): void {
		if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
			this.mediaRecorder.stop();
		}
		this.cleanup();
	}

	private initializeRecorder(stream: MediaStream): void {
		const options: MediaRecorderOptions = {};
		
		if (MediaRecorder.isTypeSupported(AudioMimeType.WAV)) {
			options.mimeType = AudioMimeType.WAV;
		} else if (MediaRecorder.isTypeSupported(AudioMimeType.WEBM_OPUS)) {
			options.mimeType = AudioMimeType.WEBM_OPUS;
		} else if (MediaRecorder.isTypeSupported(AudioMimeType.WEBM)) {
			options.mimeType = AudioMimeType.WEBM;
		} else if (MediaRecorder.isTypeSupported(AudioMimeType.MP4)) {
			options.mimeType = AudioMimeType.MP4;
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

	private cleanup(): void {
		if (this.stream) {
			this.stream.getTracks().forEach(track => track.stop());
			this.stream = null;
		}
		this.mediaRecorder = null;
		this.audioChunks = [];
		this.recordingState = false;
	}
}

export async function convertToWav(audioBlob: Blob): Promise<Blob> {
	try {
		if (audioBlob.type.includes('wav')) {
			return audioBlob;
		}

		const arrayBuffer = await audioBlob.arrayBuffer();
		
		const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
		
		const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
		
		const wavBlob = audioBufferToWav(audioBuffer);
		
		audioContext.close();
		
		return wavBlob;
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
	
	let offset = 44;
	for (let i = 0; i < length; i++) {
		for (let channel = 0; channel < numberOfChannels; channel++) {
			const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
			view.setInt16(offset, sample * 0x7FFF, true);
			offset += 2;
		}
	}
	
	return new Blob([arrayBuffer], { type: AudioMimeType.WAV });
}

export function createAudioFile(audioBlob: Blob, filename?: string): File {
	const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
	const extension = audioBlob.type.includes('wav') ? 'wav' : 'mp3';
	const defaultFilename = `recording-${timestamp}.${extension}`;
	
	return new File([audioBlob], filename || defaultFilename, {
		type: audioBlob.type,
		lastModified: Date.now()
	});
}

export function isAudioRecordingSupported(): boolean {
	return !!(
		typeof navigator !== 'undefined' &&
		navigator.mediaDevices &&
		typeof navigator.mediaDevices.getUserMedia === 'function' &&
		typeof window !== 'undefined' &&
		window.MediaRecorder
	);
}
