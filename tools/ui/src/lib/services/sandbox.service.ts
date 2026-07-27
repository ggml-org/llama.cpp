import {
	NEWLINE,
	SANDBOX_EMPTY_OUTPUT,
	SANDBOX_OUTPUT_MAX_CHARS,
	SANDBOX_TIMEOUT_MS_DEFAULT,
	SANDBOX_TIMEOUT_MS_MAX,
	SANDBOX_TOOL_NAME,
	SANDBOX_TRUNCATION_NOTICE
} from '$lib/constants';
import { buildSandboxHarness } from './sandbox-harness';
import { config } from '$lib/stores/settings.svelte';
import type { ToolExecutionResult } from '$lib/types';

/** Cached harnesses keyed by whether nerdamer is included. */
const harnessCache: Record<string, string> = {};

/**
 * Build the sandbox harness. When symbolic math is enabled, loads the
 * nerdamer prelude lazily; otherwise builds a plain harness with an empty
 * prelude. Cached per variant so toggling the setting is instant.
 */
async function getHarness(): Promise<string> {
	const enabled = !!config().symbolicMathEnabled;
	const key = enabled ? 'nerdamer' : 'plain';
	if (!harnessCache[key]) {
		if (enabled) {
			const { default: nerdamerJs } = await import('virtual:nerdamer');
			harnessCache[key] = buildSandboxHarness(nerdamerJs);
		} else {
			harnessCache[key] = buildSandboxHarness('');
		}
	}
	return harnessCache[key];
}

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

	let content = lines.join(NEWLINE);
	if (!content) content = SANDBOX_EMPTY_OUTPUT;
	if (content.length > SANDBOX_OUTPUT_MAX_CHARS) {
		content = `${content.slice(0, SANDBOX_OUTPUT_MAX_CHARS)}${NEWLINE}${SANDBOX_TRUNCATION_NOTICE}`;
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
	static async executeTool(
		toolName: string,
		params: Record<string, unknown>,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		if (toolName !== SANDBOX_TOOL_NAME) {
			return { content: `Unknown frontend tool: ${toolName}`, isError: true };
		}

		const code = typeof params.code === 'string' ? params.code : '';
		if (!code) {
			return { content: 'Missing required parameter: code', isError: true };
		}

		const harness = await getHarness();

		const requested = Number(params.timeout_ms);
		const timeoutMs =
			Number.isFinite(requested) && requested > 0
				? Math.min(requested, SANDBOX_TIMEOUT_MS_MAX)
				: SANDBOX_TIMEOUT_MS_DEFAULT;

		return new Promise<ToolExecutionResult>((resolve, reject) => {
			const iframe = document.createElement('iframe');
			iframe.setAttribute('sandbox', 'allow-scripts');
			iframe.style.display = 'none';
			iframe.srcdoc = harness;

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
