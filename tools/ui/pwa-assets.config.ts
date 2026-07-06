import {
	combinePresetAndAppleSplashScreens,
	defineConfig,
	minimal2023Preset
} from '@vite-pwa/assets-generator/config';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
	THEME_COLORS,
	PWA_GENERATOR_DEVICES,
	PWA_ASSET_GENERATOR,
	FAVICON_COLORS
} from './src/lib/constants/pwa';
import { SplashOrientation } from './src/lib/enums/splash.enums';
import { writeThemeFavicons } from './scripts/favicon-colorize';

writeThemeFavicons(FAVICON_COLORS.LIGHT, FAVICON_COLORS.DARK, {
	padding: PWA_ASSET_GENERATOR.FAVICON_PADDING
});

export default defineConfig({
	headLinkOptions: {
		preset: PWA_ASSET_GENERATOR.LINK_PRESET
	},
	preset: combinePresetAndAppleSplashScreens(
		{
			...minimal2023Preset,
			// tiny margin so favicon.ico / pwa-*.png breathe inside the canvas
			transparent: {
				...minimal2023Preset.transparent,
				padding: PWA_ASSET_GENERATOR.FAVICON_PADDING
			}
		},
		{
			padding: PWA_ASSET_GENERATOR.SPLASH_PADDING,
			resizeOptions: {
				background: THEME_COLORS.BACKGROUND_LIGHT,
				fit: PWA_ASSET_GENERATOR.FIT_MODE
			},
			darkResizeOptions: {
				background: THEME_COLORS.BACKGROUND_DARK,
				fit: PWA_ASSET_GENERATOR.FIT_MODE
			},
			darkImageResolver: async (imageName: string) => {
				if (imageName.endsWith('favicon.svg')) {
					return readFileSync(resolve('static/favicon-dark.svg'));
				}
			},
			linkMediaOptions: {
				log: true,
				addMediaScreen: PWA_ASSET_GENERATOR.ADD_MEDIA_SCREEN,
				basePath: PWA_ASSET_GENERATOR.BASE_PATH,
				xhtml: PWA_ASSET_GENERATOR.XHTML
			},
			png: {
				compressionLevel: PWA_ASSET_GENERATOR.PNG_COMPRESSION_LEVEL,
				quality: PWA_ASSET_GENERATOR.PNG_QUALITY
			},
			name: (landscape, size, dark) => {
				const orientation = landscape ? SplashOrientation.LANDSCAPE : SplashOrientation.PORTRAIT;
				const darkPrefix = dark ? PWA_ASSET_GENERATOR.DARK_PREFIX : '';
				return `apple-splash-${orientation}-${darkPrefix}${size.width}x${size.height}.png`;
			}
		},
		PWA_GENERATOR_DEVICES
	),
	images: ['static/favicon.svg']
});
