import { defineConfig } from '@vite-pwa/assets-generator/config';
import { FAVICON_COLORS } from './src/lib/constants/pwa';
import { writeThemeFavicons } from './scripts/favicon-colorize';

writeThemeFavicons(FAVICON_COLORS.LIGHT, FAVICON_COLORS.DARK);

export default defineConfig({
	headLinkOptions: {
		preset: '2023'
	},
	preset: {
		transparent: {
			sizes: [],
			favicons: [[48, 'favicon-dark.ico']]
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
