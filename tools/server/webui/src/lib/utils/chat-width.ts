export const DEFAULT_WIDTH = 'max-w-[48rem]';

export const MIN_CUSTOM_WIDTH = 300;
export const MAX_CUSTOM_WIDTH = 10000;

export const AUTO_WIDTH_CLASSES = `
  max-w-[48rem] 
  md:max-w-[60rem] 
  xl:max-w-[70rem] 
  2xl:max-w-[80rem] 
  3xl:max-w-[90rem] 
  4xl:max-w-[100rem] 
  5xl:max-w-[150rem]
`;

export const CUSTOM_WIDTH_PRESETS = {
	xs: 480,
	sm: 600,
	md: 768,
	lg: 960,
	xl: 1152,
	'2xl': 1280,
	'3xl': 1440,
	'4xl': 1600,
	'5xl': 1920,
	'6xl': 2304,
	'7xl': 3072
} as const;

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
