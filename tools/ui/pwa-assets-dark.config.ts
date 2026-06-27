import { defineConfig } from '@vite-pwa/assets-generator/config';
import { FAVICON_COLORS, PWA_ASSET_GENERATOR } from './src/lib/constants/pwa';
import { writeThemeFavicons } from './scripts/favicon-colorize';

writeThemeFavicons(FAVICON_COLORS.LIGHT, FAVICON_COLORS.DARK, {
	padding: PWA_ASSET_GENERATOR.FAVICON_PADDING
});

export default defineConfig({
	headLinkOptions: {
		preset: '2023'
	},
	preset: {
		transparent: {
			sizes: [],
			favicons: [[48, 'favicon-dark.ico']],
			padding: PWA_ASSET_GENERATOR.FAVICON_PADDING
		},
		maskable: {
			sizes: []
		},
		apple: {
			sizes: []
		}
	},
	images: ['static/favicon-dark.svg']
});
