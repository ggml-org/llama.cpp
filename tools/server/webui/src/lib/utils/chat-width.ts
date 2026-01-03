import { AUTO_WIDTH_CLASSES, CUSTOM_WIDTH_PRESETS, DEFAULT_WIDTH } from '$lib/constants/chat-width';

export type CustomWidthPreset = keyof typeof CUSTOM_WIDTH_PRESETS;

/**
 * Get the appropriate width configuration based on settings
 * @param autoChatWidth - Whether automatic responsive width is enabled
 * @param customChatWidth - Custom width setting (preset key or pixel value)
 */
export function getChatWidth(
	autoChatWidth: boolean,
	customChatWidth: string
): { class: string; style?: string } {
	if (autoChatWidth) {
		return { class: AUTO_WIDTH_CLASSES };
	}

	if (customChatWidth) {
		if (customChatWidth in CUSTOM_WIDTH_PRESETS) {
			const pixelValue = CUSTOM_WIDTH_PRESETS[customChatWidth as CustomWidthPreset];
			return { class: '', style: `max-width: ${pixelValue}px` };
		}

		const numValue = Number(customChatWidth);
		if (!isNaN(numValue) && numValue > 0) {
			return { class: '', style: `max-width: ${numValue}px` };
		}
	}

	return { class: DEFAULT_WIDTH };
}
