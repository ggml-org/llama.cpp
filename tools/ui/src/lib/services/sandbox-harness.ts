import { NEWLINE } from '$lib/constants';
import WORKER_SHIM from './sandbox-worker.js?raw';

/**
 * CSP for the harness document, inherited by the blob worker. connect-src
 * falls back to default-src, removing network egress for model and vendored
 * code. 'unsafe-eval' is required by the worker's AsyncFunction constructor,
 * 'unsafe-inline' by the inline script below, worker-src by the blob worker.
 */
const HARNESS_CSP = `default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; worker-src blob:`;

/**
 * Harness loaded as srcdoc into a sandboxed iframe (allow-scripts only).
 * The opaque origin is the security boundary: no access to the app origin,
 * its storage or its API. The harness spawns a worker so model code never
 * runs on a main thread, which makes the parent timeout enforceable by
 * removing the iframe. The prelude runs in the worker before the shim,
 * exposing globals such as `nerdamer` to model code.
 */
export function buildSandboxHarness(preludeJs: string): string {
	return `<!doctype html><meta http-equiv="Content-Security-Policy" content="${HARNESS_CSP}"><script>
const SHIM = ${JSON.stringify(preludeJs + NEWLINE + WORKER_SHIM)};
addEventListener('message', (event) => {
	const respond = (payload) => parent.postMessage(payload, '*');
	let worker;
	try {
		worker = new Worker(URL.createObjectURL(new Blob([SHIM], { type: 'text/javascript' })));
	} catch (err) {
		respond({ logs: [], result: null, error: 'Worker creation failed: ' + err });
		return;
	}
	worker.onmessage = (msg) => respond(msg.data);
	worker.onerror = (err) => respond({ logs: [], result: null, error: String(err.message || err) });
	worker.postMessage({ code: event.data.code });
});
</script>`;
}
