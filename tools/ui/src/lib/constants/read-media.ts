import {
	BuiltInTool,
	JsonSchemaType,
	MimeTypeAudio,
	MimeTypeImage,
	ToolCallType
} from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const READ_MEDIA_TOOL_NAME = BuiltInTool.READ_MEDIA;

// header lines of the tool result, parsed back by the read_media renderer
export const PREFIX_FILE = 'File: ';
export const PREFIX_SIZE = 'Size: ';
export const PREFIX_MIME = 'MIME: ';

/** Byte count of the `Size: ` header line, e.g. `Size: 12345 bytes` -> capture group 1 is `12345`. */
export const READ_MEDIA_SIZE_REGEX = new RegExp(`^${PREFIX_SIZE}\\s*(\\d+)\\s*bytes`);

/** Image extensions the tool accepts. The server decodes images with stb_image, which has no webp or tiff. */
export const READ_MEDIA_IMAGE_MIME: Record<string, string> = {
	gif: MimeTypeImage.GIF,
	jpeg: MimeTypeImage.JPEG,
	jpg: MimeTypeImage.JPEG,
	png: MimeTypeImage.PNG
} as const;

/** Audio extensions the tool accepts. The `input_audio` API only takes wav and mp3. */
export const READ_MEDIA_AUDIO_MIME: Record<string, string> = {
	mp3: MimeTypeAudio.MP3_MPEG,
	wav: MimeTypeAudio.WAV
} as const;

/**
 * Build the read_media tool definition for the modalities the active model has.
 * At least one of the two flags must be true, otherwise the tool is not offered
 * at all - a model that cannot see or hear has nothing to do with the bytes.
 */
export function buildReadMediaToolDefinition(
	supportsVision: boolean,
	supportsAudio: boolean
): OpenAIToolDefinition {
	const kinds: string[] = [];

	if (supportsVision) kinds.push(`images (${Object.keys(READ_MEDIA_IMAGE_MIME).join(', ')})`);

	if (supportsAudio) kinds.push(`audio (${Object.keys(READ_MEDIA_AUDIO_MIME).join(', ')})`);

	return {
		function: {
			description: `Read a media file and attach it to the conversation so it can be perceived directly. Supports ${kinds.join(' and ')}.`,
			name: READ_MEDIA_TOOL_NAME,
			parameters: {
				properties: {
					path: {
						description: 'Path to the media file',
						type: JsonSchemaType.STRING
					}
				},
				required: ['path'],
				type: JsonSchemaType.OBJECT
			}
		},
		type: ToolCallType.FUNCTION
	};
}
