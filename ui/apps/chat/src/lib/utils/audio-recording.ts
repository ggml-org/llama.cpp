import { MimeTypeAudio } from '#lib/enums/index.ts';

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
		lastModified: Date.now(),
		type: audioBlob.type
	});
}
