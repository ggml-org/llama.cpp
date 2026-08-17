import { FuseV1Options, FuseVersion } from '@electron/fuses';
import { MakerDMG } from '@electron-forge/maker-dmg';
import { MakerSquirrel } from '@electron-forge/maker-squirrel';
import { MakerZIP } from '@electron-forge/maker-zip';
import { FusesPlugin } from '@electron-forge/plugin-fuses';
import { VitePlugin } from '@electron-forge/plugin-vite';
import type { ForgeConfig } from '@electron-forge/shared-types';

const { LLAMA_SERVER_PATH, LLAMA_SHARED_LIBS, OUT_DIR } = process.env;
const config: ForgeConfig = {
	...(OUT_DIR ? { outDir: OUT_DIR } : {}),
	makers: [
		new MakerSquirrel(
			{
				name: 'llama',
				setupIcon: 'src/assets/icon.ico'
			},
			['win32']
		),
		new MakerZIP({}, ['darwin']),
		new MakerDMG(
			{
				icon: 'src/assets/icon.icns'
			},
			['darwin']
		)
	],
	packagerConfig: {
		asar: true,
		extraResource: [
			...(LLAMA_SERVER_PATH ? [LLAMA_SERVER_PATH] : []),
			...(LLAMA_SHARED_LIBS ? LLAMA_SHARED_LIBS.split(';') : [])
		],
		icon: 'src/assets/icon'
	},
	plugins: [
		new VitePlugin({
			build: [
				{
					config: 'vite.main.config.mts',
					entry: 'src/main.ts',
					target: 'main'
				}
			],
			renderer: []
		}),
		new FusesPlugin({
			[FuseV1Options.EnableCookieEncryption]: true,
			[FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
			[FuseV1Options.EnableNodeCliInspectArguments]: false,
			[FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
			[FuseV1Options.GrantFileProtocolExtraPrivileges]: false,
			[FuseV1Options.LoadBrowserProcessSpecificV8Snapshot]: false,
			[FuseV1Options.OnlyLoadAppFromAsar]: true,
			[FuseV1Options.RunAsNode]: false,
			[FuseV1Options.WasmTrapHandlers]: true,
			strictlyRequireAllFuses: true,
			version: FuseVersion.V1
		})
	],
	rebuildConfig: {}
};

export default config;
