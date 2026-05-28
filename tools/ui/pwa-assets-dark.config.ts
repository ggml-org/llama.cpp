import { defineConfig } from '@vite-pwa/assets-generator/config';

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
