import type { JsonRpcMessage } from '../types';

export interface MCPTransport {
	start(): Promise<void>;
	stop(): Promise<void>;
	send(message: JsonRpcMessage): Promise<void>;
	onMessage(handler: (message: JsonRpcMessage) => void): void;
}
