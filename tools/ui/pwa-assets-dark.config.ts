import { writeThemeFavicons } from './scripts/favicon-colorize';
import { FAVICON_COLORS, PWA_ASSET_GENERATOR } from './src/lib/constants/pwa.constants';
import { defineConfig } from '@vite-pwa/assets-generator/config';

writeThemeFavicons(FAVICON_COLORS.LIGHT, FAVICON_COLORS.DARK, {
	padding: PWA_ASSET_GENERATOR.FAVICON_PADDING
});

export default defineConfig({
	headLinkOptions: {
		preset: '2023'
	},
	images: ['static/favicon-dark.svg'],
	preset: {
		apple: {
			sizes: []
		},
		maskable: {
			sizes: []
		},
		transparent: {
			favicons: [[48, 'favicon-dark.ico']],
			padding: PWA_ASSET_GENERATOR.FAVICON_PADDING,
			sizes: []
		}
	}
});
