import WORKER_SHIM from './sandbox-worker.js?raw';

/**
 * Harness loaded as srcdoc into a sandboxed iframe (allow-scripts only).
 * The opaque origin is the security boundary: no access to the app origin,
 * its storage or its API. The harness spawns a worker so model code never
 * runs on a main thread, which makes the parent timeout enforceable by
 * removing the iframe.
 */
export const SANDBOX_HARNESS_HTML = `<!doctype html><script>
const SHIM = ${JSON.stringify(WORKER_SHIM)};
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
