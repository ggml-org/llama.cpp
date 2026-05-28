import {
	combinePresetAndAppleSplashScreens,
	defineConfig,
	minimal2023Preset
} from '@vite-pwa/assets-generator/config';

export default defineConfig({
	headLinkOptions: {
		preset: '2023'
	},
	preset: combinePresetAndAppleSplashScreens(
		minimal2023Preset,
		{
			padding: 0.75,
			resizeOptions: { background: 'white', fit: 'contain' },
			darkResizeOptions: { background: '#111111', fit: 'contain' },
			linkMediaOptions: {
				log: true,
				addMediaScreen: true,
				basePath: '/',
				xhtml: false
			},
			png: {
				compressionLevel: 9,
				quality: 60
			},
			name: (landscape, size, dark) => {
				return `apple-splash-${landscape ? 'landscape' : 'portrait'}-${dark ? 'dark-' : ''}${size.width}x${size.height}.png`;
			}
		},
		[
			// iPhones (current generation + one previous)
			'iPhone 13',
			'iPhone 13 Pro',
			'iPhone 13 Pro Max',
			'iPhone 14',
			'iPhone 14 Plus',
			'iPhone 14 Pro',
			'iPhone 14 Pro Max',
			'iPhone 15',
			'iPhone 15 Plus',
			'iPhone 15 Pro',
			'iPhone 15 Pro Max',
			'iPhone 16',
			'iPhone 16 Plus',
			'iPhone 16 Pro',
			'iPhone 16 Pro Max',
			'iPhone 16e',
			'iPhone SE 4"',
			'iPhone SE 4.7"',
			// iPads
			'iPad 11"',
			'iPad Air 10.9"',
			'iPad Air 11"',
			'iPad Air 13"',
			'iPad Pro 11"',
			'iPad Pro 12.9"',
			'iPad mini 8.3"'
		]
	),
	images: ['static/favicon.svg']
});
