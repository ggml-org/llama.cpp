import { defineConfig } from '@vite-pwa/assets-generator/config';

// Custom preset: same as minimal2023 but with white background for pwa icons
const preset = {
	transparent: {
		sizes: [64, 192, 512],
		favicons: [[48, 'favicon.ico']],
		resizeOptions: {
			fit: 'contain',
			background: 'white'
		}
	},
	maskable: {
		sizes: [512]
	},
	apple: {
		sizes: [180]
	}
};

export default defineConfig({
	preset,
	images: ['static/logo.svg']
});
