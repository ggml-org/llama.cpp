import { getAudioInputFormat } from './audio-format';
import { capImageDataURLSize } from './cap-img-size';
import { AttachmentType, ContentPartType } from '#lib/enums/index.ts';
import type { AudioInputFormat } from '#lib/types/api.js';
import type {
	DatabaseMessageExtra,
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

/** Modalities a model takes, for the media of a tool result. */
export interface ToolMediaInput {
	audio: boolean;
	vision: boolean;
}

/** An attachment's identity: its name and its bytes. */
function attachmentKey(extra: DatabaseMessageExtra): string {
	if (extra.type === AttachmentType.IMAGE) return `${extra.name}\0${extra.base64Url}`;

	if (extra.type === AttachmentType.AUDIO) return `${extra.name}\0${extra.base64Data}`;

	return `${extra.type}\0${extra.name}`;
}

/**
 * The attachments without the ones repeated with the same name and bytes. Tool results
 * once stored their media twice; the copy adds nothing to the trace or the context.
 */
export function withoutRepeatedAttachments(extras: DatabaseMessageExtra[]): DatabaseMessageExtra[] {
	const seen = new Set<string>();

	return extras.filter((extra) => {
		const key = attachmentKey(extra);

		if (seen.has(key)) return false;

		seen.add(key);

		return true;
	});
}

/**
 * The media of a tool result as content parts, only what the model can read. The run that
 * calls the tool and every later request build them here, so the server's cache stays valid.
 */
export async function toolResultMediaParts(
	extras: DatabaseMessageExtra[],
	input: ToolMediaInput,
	maxMegapixels: number
): Promise<(AudioContentPart | ImageContentPart)[]> {
	const parts: (AudioContentPart | ImageContentPart)[] = [];

	for (const extra of withoutRepeatedAttachments(extras)) {
		if (extra.type === AttachmentType.AUDIO && input.audio) {
			parts.push(audioContentPart(extra));
		} else if (extra.type === AttachmentType.IMAGE && input.vision) {
			parts.push(
				// a tool can return a format the cap cannot draw: send it as it is
				await imageContentPart(extra, maxMegapixels).catch(() => ({
					image_url: { url: extra.base64Url },
					type: ContentPartType.IMAGE_URL as const
				}))
			);
		}
	}

	return parts;
}
