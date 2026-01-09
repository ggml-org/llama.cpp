/**
 * MCP-specific error class with error code and optional data.
 */
export class MCPError extends Error {
	code: number;
	data?: unknown;

	constructor(message: string, code: number, data?: unknown) {
		super(message);
		this.name = 'MCPError';
		this.code = code;
		this.data = data;
	}
}
