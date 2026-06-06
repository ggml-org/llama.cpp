import {
	SANDBOX_OUTPUT_MAX_CHARS,
	SANDBOX_TIMEOUT_MS_DEFAULT,
	SANDBOX_TIMEOUT_MS_MAX,
	SANDBOX_TOOL_NAME
} from '$lib/constants';
import type { ToolExecutionResult } from '$lib/types';

/**
 * Source of the worker that runs model generated code.
 * Captures console output and the value returned by the code,
 * then reports back to the iframe harness in a single message.
 */
const WORKER_SHIM = `
const logs = [];
const fmt = (value) => {
	if (typeof value === 'string') return value;
	try {
		return JSON.stringify(value);
	} catch {
		return String(value);
	}
};
const capture = (level, prefix) => (...args) => {
	logs.push(prefix + args.map(fmt).join(' '));
};
console.log = capture('log', '');
console.info = capture('info', '');
console.debug = capture('debug', '');
console.warn = capture('warn', 'warn: ');
console.error = capture('error', 'error: ');
self.onmessage = async (event) => {
	const reply = { logs, result: null, error: null };
	try {
		const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
		const value = await new AsyncFunction(event.data.code)();
		if (value !== undefined) reply.result = fmt(value);
	} catch (err) {
		reply.error = err instanceof Error ? err.stack || err.message : String(err);
	}
	self.postMessage(reply);
};
`;

/**
 * Harness loaded as srcdoc into a sandboxed iframe (allow-scripts only).
 * The opaque origin is the security boundary: no access to the app origin,
 * its storage or its API. The harness spawns a worker so model code never
 * runs on a main thread, which makes the parent timeout enforceable by
 * removing the iframe.
 */
const HARNESS_HTML = `<!doctype html><script>
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

interface SandboxReply {
	logs?: unknown;
	result?: unknown;
	error?: unknown;
}

function formatReply(reply: SandboxReply): ToolExecutionResult {
	const lines: string[] = [];

	if (Array.isArray(reply.logs)) {
		for (const line of reply.logs) lines.push(String(line));
	}

	if (reply.error != null) {
		lines.push(`Error: ${String(reply.error)}`);
	} else if (reply.result != null) {
		lines.push(`=> ${String(reply.result)}`);
	}

	let content = lines.join('\n');
	if (!content) content = '(no output)';
	if (content.length > SANDBOX_OUTPUT_MAX_CHARS) {
		content = `${content.slice(0, SANDBOX_OUTPUT_MAX_CHARS)}\n[output truncated]`;
	}

	return { content, isError: reply.error != null };
}

export class SandboxService {
	/**
	 * Execute a frontend sandbox tool call and return its output.
	 * One disposable iframe per execution, removed on completion,
	 * timeout or abort. Removing the iframe terminates the worker
	 * at the browser level, so runaway code cannot outlive it.
	 */
	static executeTool(
		toolName: string,
		params: Record<string, unknown>,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		if (toolName !== SANDBOX_TOOL_NAME) {
			return Promise.resolve({ content: `Unknown frontend tool: ${toolName}`, isError: true });
		}

		const code = typeof params.code === 'string' ? params.code : '';
		if (!code) {
			return Promise.resolve({ content: 'Missing required parameter: code', isError: true });
		}

		const requested = Number(params.timeout_ms);
		const timeoutMs =
			Number.isFinite(requested) && requested > 0
				? Math.min(requested, SANDBOX_TIMEOUT_MS_MAX)
				: SANDBOX_TIMEOUT_MS_DEFAULT;

		return new Promise<ToolExecutionResult>((resolve, reject) => {
			const iframe = document.createElement('iframe');
			iframe.setAttribute('sandbox', 'allow-scripts');
			iframe.style.display = 'none';
			iframe.srcdoc = HARNESS_HTML;

			let settled = false;

			const cleanup = () => {
				settled = true;
				clearTimeout(timer);
				window.removeEventListener('message', onMessage);
				signal?.removeEventListener('abort', onAbort);
				iframe.remove();
			};

			const finish = (result: ToolExecutionResult) => {
				if (settled) return;
				cleanup();
				resolve(result);
			};

			const onAbort = () => {
				if (settled) return;
				cleanup();
				reject(new DOMException('Sandbox execution aborted', 'AbortError'));
			};

			const onMessage = (event: MessageEvent) => {
				if (event.source !== iframe.contentWindow) return;
				finish(formatReply((event.data ?? {}) as SandboxReply));
			};

			const timer = setTimeout(
				() => finish({ content: `Execution timed out after ${timeoutMs} ms`, isError: true }),
				timeoutMs
			);

			window.addEventListener('message', onMessage);
			signal?.addEventListener('abort', onAbort);
			iframe.onload = () => iframe.contentWindow?.postMessage({ code }, '*');
			document.body.appendChild(iframe);
		});
	}
}
