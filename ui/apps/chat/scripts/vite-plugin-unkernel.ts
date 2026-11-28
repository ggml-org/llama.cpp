import type { Plugin } from 'vite';

const UNKERNEL_WEB_ENTRY = '/node_modules/unkernel/dist/web.mjs';
const BUNDLED_KERNEL_URL = 'new URL("./unkernel.wasm", import.meta.url)';
const NO_KERNEL_URL = 'new URL("about:blank")';

/**
 * Keep unkernel's kernel (GPL-2.0, 10 MB) out of the build. The SDK's web entry names
 * the kernel shipped next to it, which Vite would emit as an asset of the app, and so
 * of llama-server; Llama Chat loads the kernel of its Linux image's release instead
 * (LINUX_KERNEL_URL). The build fails if the entry stops naming its kernel this way.
 */
export function unkernelPlugin(): Plugin {
	return {
		enforce: 'pre',
		name: 'llama:unkernel',
		transform(code, id) {
			if (!id.split('?')[0].endsWith(UNKERNEL_WEB_ENTRY)) return null;

			if (!code.includes(BUNDLED_KERNEL_URL)) {
				this.error(`${UNKERNEL_WEB_ENTRY} no longer names its kernel as ${BUNDLED_KERNEL_URL}`);
			}

			return code.replace(BUNDLED_KERNEL_URL, NO_KERNEL_URL);
		}
	};
}
