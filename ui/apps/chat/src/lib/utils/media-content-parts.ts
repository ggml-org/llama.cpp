import { getAudioInputFormat } from './audio-format';
import { capImageDataURLSize } from './cap-img-size';
import { ContentPartType } from '#lib/enums/index.ts';
import type { AudioInputFormat } from '#lib/types/api.js';
import type {
	DatabaseMessageExtraAudioFile,
	DatabaseMessageExtraImageFile
} from '#lib/types/database.js';

export interface ImageContentPart {
	image_url: { url: string };
	type: ContentPartType.IMAGE_URL;
}

export interface AudioContentPart {
	input_audio: { data: string; format: AudioInputFormat };
	type: ContentPartType.INPUT_AUDIO;
}

/**
 * An image as the model receives it: capped to the max resolution, with the jpeg exif orientation baked in.
 * User attachments and tool results both go through here, so they cost the same.
 */
export async function imageContentPart(
	image: DatabaseMessageExtraImageFile,
	maxMegapixels: number
): Promise<ImageContentPart> {
	return {
		image_url: { url: await capImageDataURLSize(image.base64Url, maxMegapixels) },
		type: ContentPartType.IMAGE_URL
	};
}

export function audioContentPart(audio: DatabaseMessageExtraAudioFile): AudioContentPart {
	return {
		input_audio: { data: audio.base64Data, format: getAudioInputFormat(audio.mimeType) },
		type: ContentPartType.INPUT_AUDIO
	};
}
