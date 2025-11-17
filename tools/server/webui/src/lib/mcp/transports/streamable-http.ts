import type { JsonRpcMessage } from '$lib/mcp/types';
import type { MCPTransport } from './types';

export type StreamableHttpTransportOptions = {
	url: string;
	headers?: Record<string, string>;
	credentials?: RequestCredentials;
	protocolVersion?: string;
	sessionId?: string;
};

export class StreamableHttpTransport implements MCPTransport {
	private handler: ((message: JsonRpcMessage) => void) | null = null;
	private activeSessionId: string | undefined;

	constructor(private readonly options: StreamableHttpTransportOptions) {}

	async start(): Promise<void> {
		this.activeSessionId = this.options.sessionId ?? undefined;
	}

	async stop(): Promise<void> {}

	async send(message: JsonRpcMessage): Promise<void> {
		return this.dispatch(message);
	}

	onMessage(handler: (message: JsonRpcMessage) => void): void {
		this.handler = handler;
	}

	private async dispatch(message: JsonRpcMessage): Promise<void> {
		const headers: Record<string, string> = {
			'Content-Type': 'application/json',
			Accept: 'application/json, text/event-stream',
			...(this.options.headers ?? {})
		};

		if (this.activeSessionId) {
			headers['Mcp-Session-Id'] = this.activeSessionId;
		}

		if (this.options.protocolVersion) {
			headers['MCP-Protocol-Version'] = this.options.protocolVersion;
		}

		const credentialsOption =
			this.options.credentials ?? (this.activeSessionId ? 'include' : 'same-origin');
		const response = await fetch(this.options.url, {
			method: 'POST',
			headers,
			body: JSON.stringify(message),
			credentials: credentialsOption
		});

		const sessionHeader = response.headers.get('mcp-session-id');
		if (sessionHeader) {
			this.activeSessionId = sessionHeader;
		}

		if (!response.ok) {
			const errorBody = await response.text().catch(() => '');
			throw new Error(
				`Failed to send MCP request over Streamable HTTP (${response.status} ${response.statusText}): ${errorBody}`
			);
		}

		const contentType = response.headers.get('content-type') ?? '';

		if (contentType.includes('application/json')) {
			const payload = (await response.json()) as JsonRpcMessage;
			this.handler?.(payload);
			return;
		}

		if (contentType.includes('text/event-stream') && response.body) {
			const reader = response.body.getReader();
			await this.consume(reader);
			return;
		}

		if (response.status >= 400) {
			const bodyText = await response.text().catch(() => '');
			throw new Error(
				`Unexpected MCP Streamable HTTP response (${response.status}): ${bodyText || 'no body'}`
			);
		}
	}

	private async consume(reader: ReadableStreamDefaultReader<Uint8Array>): Promise<void> {
		const decoder = new TextDecoder('utf-8');
		let buffer = '';

		try {
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });

				const parts = buffer.split('\n\n');
				buffer = parts.pop() ?? '';

				for (const part of parts) {
					if (!part.startsWith('data: ')) {
						continue;
					}
					const payload = part.slice(6);
					if (!payload || payload === '[DONE]') {
						continue;
					}

					try {
						const message = JSON.parse(payload) as JsonRpcMessage;
						this.handler?.(message);
					} catch (error) {
						console.error('[MCP][Streamable HTTP] Failed to parse JSON payload:', error);
					}
				}
			}
		} catch (error) {
			if ((error as Error)?.name === 'AbortError') {
				return;
			}
			throw error;
		} finally {
			reader.releaseLock();
		}
	}
}
