import type {
	JsonRpcId,
	JsonRpcMessage,
	JsonRpcNotification,
	JsonRpcRequest,
	JsonRpcResponse
} from './types';

export class JsonRpcProtocol {
	static createRequest(
		id: JsonRpcId,
		method: string,
		params?: Record<string, unknown>
	): JsonRpcRequest {
		return {
			jsonrpc: '2.0',
			id,
			method,
			...(params ? { params } : {})
		};
	}

	static createNotification(method: string, params?: Record<string, unknown>): JsonRpcNotification {
		return {
			jsonrpc: '2.0',
			method,
			...(params ? { params } : {})
		};
	}

	static parseResponse(message: JsonRpcMessage): JsonRpcResponse | null {
		if (!message || typeof message !== 'object') {
			return null;
		}

		if ((message as JsonRpcResponse).jsonrpc !== '2.0') {
			return null;
		}

		if (!('id' in message)) {
			return null;
		}

		return message as JsonRpcResponse;
	}
}
