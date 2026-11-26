import { VOICE_WAV_MIME } from './constants.js';

const WAV_HEADER_BYTES = 44;

/**
 * Mono PCM16 samples as a WAV file: the format `/v1/audio/transcriptions` and the
 * chat's `input_audio` parts accept.
 */
export function encodeWav(samples: Int16Array, sampleRate: number): Blob {
	const dataSize = samples.length * 2;
	const buffer = new ArrayBuffer(WAV_HEADER_BYTES + dataSize);
	const view = new DataView(buffer);
	const writeAscii = (offset: number, text: string) => {
		for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
	};

	writeAscii(0, 'RIFF');
	view.setUint32(4, WAV_HEADER_BYTES - 8 + dataSize, true);
	writeAscii(8, 'WAVE');
	writeAscii(12, 'fmt ');
	view.setUint32(16, 16, true); // fmt chunk size
	view.setUint16(20, 1, true); // PCM
	view.setUint16(22, 1, true); // mono
	view.setUint32(24, sampleRate, true);
	view.setUint32(28, sampleRate * 2, true); // byte rate
	view.setUint16(32, 2, true); // block align
	view.setUint16(34, 16, true); // bits per sample
	writeAscii(36, 'data');
	view.setUint32(40, dataSize, true);

	for (let i = 0; i < samples.length; i++) {
		view.setInt16(WAV_HEADER_BYTES + i * 2, samples[i], true);
	}

	return new Blob([buffer], { type: VOICE_WAV_MIME });
}
