import type { JsonRpcMessage } from '$lib/mcp/types';
import type { MCPTransport } from './types';

export type WebSocketTransportOptions = {
	url: string;
	protocols?: string | string[];
	handshakeTimeoutMs?: number;
};

export type TransportMessageHandler = (message: JsonRpcMessage) => void;

function ensureWebSocket(): typeof WebSocket | null {
	if (typeof WebSocket !== 'undefined') {
		return WebSocket;
	}
	return null;
}

function arrayBufferToString(buffer: ArrayBufferLike): string {
	return new TextDecoder('utf-8').decode(new Uint8Array(buffer));
}

async function normalizePayload(data: unknown): Promise<string> {
	if (typeof data === 'string') {
		return data;
	}

	if (data instanceof ArrayBuffer) {
		return arrayBufferToString(data);
	}

	if (ArrayBuffer.isView(data)) {
		return arrayBufferToString(data.buffer);
	}

	if (typeof Blob !== 'undefined' && data instanceof Blob) {
		return await data.text();
	}

	throw new Error('Unsupported WebSocket message payload type');
}

export class WebSocketTransport implements MCPTransport {
	private socket: WebSocket | null = null;
	private handler: TransportMessageHandler | null = null;
	private openPromise: Promise<void> | null = null;
	private reconnectAttempts = 0;
	private readonly maxReconnectAttempts = 5;
	private readonly reconnectDelay = 1_000;
	private isReconnecting = false;
	private shouldAttemptReconnect = true;

	constructor(private readonly options: WebSocketTransportOptions) {}

	start(): Promise<void> {
		if (this.openPromise) {
			return this.openPromise;
		}

		this.shouldAttemptReconnect = true;

		this.openPromise = new Promise((resolve, reject) => {
			const WebSocketImpl = ensureWebSocket();
			if (!WebSocketImpl) {
				this.openPromise = null;
				reject(new Error('WebSocket is not available in this environment'));
				return;
			}

			let handshakeTimeout: ReturnType<typeof setTimeout> | undefined;
			const socket = this.options.protocols
				? new WebSocketImpl(this.options.url, this.options.protocols)
				: new WebSocketImpl(this.options.url);

			const cleanup = () => {
				if (!socket) return;
				socket.onopen = null;
				socket.onclose = null;
				socket.onerror = null;
				socket.onmessage = null;
				if (handshakeTimeout) {
					clearTimeout(handshakeTimeout);
					handshakeTimeout = undefined;
				}
			};

			const fail = (error: unknown) => {
				cleanup();
				this.openPromise = null;
				reject(error instanceof Error ? error : new Error('WebSocket connection error'));
			};

			socket.onopen = () => {
				cleanup();
				this.socket = socket;
				this.reconnectAttempts = 0;
				this.attachMessageHandler();
				this.attachCloseHandler(socket);
				resolve();
				this.openPromise = null;
			};

			socket.onerror = (event) => {
				const error = event instanceof Event ? new Error('WebSocket connection error') : event;
				fail(error);
			};

			socket.onclose = (event) => {
				if (!this.socket) {
					fail(new Error(`WebSocket closed before opening (code: ${event.code})`));
				}
			};

			if (this.options.handshakeTimeoutMs) {
				handshakeTimeout = setTimeout(() => {
					if (!this.socket) {
						try {
							socket.close();
						} catch (error) {
							console.warn('[MCP][Transport] Failed to close socket after timeout:', error);
						}
						fail(new Error('WebSocket handshake timed out'));
					}
				}, this.options.handshakeTimeoutMs);
			}
		});

		return this.openPromise;
	}

	async send(message: JsonRpcMessage): Promise<void> {
		if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
			throw new Error('WebSocket transport is not connected');
		}
		this.socket.send(JSON.stringify(message));
	}

	async stop(): Promise<void> {
		this.shouldAttemptReconnect = false;
		this.reconnectAttempts = 0;
		this.isReconnecting = false;

		const socket = this.socket;
		if (!socket) {
			this.openPromise = null;
			return;
		}

		await new Promise<void>((resolve) => {
			const onClose = () => {
				socket.removeEventListener('close', onClose);
				resolve();
			};
			socket.addEventListener('close', onClose);
			try {
				socket.close();
			} catch (error) {
				socket.removeEventListener('close', onClose);
				console.warn('[MCP][Transport] Failed to close WebSocket:', error);
				resolve();
			}
		});

		this.socket = null;
		this.openPromise = null;
	}

	onMessage(handler: TransportMessageHandler): void {
		this.handler = handler;
		this.attachMessageHandler();
	}

	private attachMessageHandler(): void {
		if (!this.socket) {
			return;
		}

		this.socket.onmessage = (event: MessageEvent) => {
			const payload = event.data;
			void (async () => {
				try {
					const text = await normalizePayload(payload);
					const parsed = JSON.parse(text);
					this.handler?.(parsed as JsonRpcMessage);
				} catch (error) {
					console.error('[MCP][Transport] Failed to handle message:', error);
				}
			})();
		};
	}

	private attachCloseHandler(socket: WebSocket): void {
		socket.onclose = (event) => {
			this.socket = null;

			if (event.code === 1000 || !this.shouldAttemptReconnect) {
				return;
			}

			console.warn('[MCP][WebSocket] Connection closed unexpectedly, attempting reconnect');
			void this.reconnect();
		};
	}

	private async reconnect(): Promise<void> {
		if (
			this.isReconnecting ||
			this.reconnectAttempts >= this.maxReconnectAttempts ||
			!this.shouldAttemptReconnect
		) {
			return;
		}

		this.isReconnecting = true;
		this.reconnectAttempts++;

		const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
		await new Promise((resolve) => setTimeout(resolve, delay));

		try {
			this.openPromise = null;
			await this.start();
			this.reconnectAttempts = 0;
			console.log('[MCP][WebSocket] Reconnected successfully');
		} catch (error) {
			console.error('[MCP][WebSocket] Reconnection failed:', error);
		} finally {
			this.isReconnecting = false;
			if (
				!this.socket &&
				this.shouldAttemptReconnect &&
				this.reconnectAttempts < this.maxReconnectAttempts
			) {
				void this.reconnect();
			}
		}
	}
}
