/** Number of trailing characters to keep visible when partially redacting mcp-session-id */
const MCP_SESSION_ID_VISIBLE_CHARS = 5;

/** HTTP header handling for API and MCP requests. */
export const HEADERS = {
	/** Canonical casing for the Authorization header (RFC 7235) */
	AUTHORIZATION: 'Authorization',
	/** Bearer scheme prefix used for Authorization headers (RFC 6750) */
	BEARER: 'Bearer ',
	/** Content-Type HTTP header name */
	CONTENT_TYPE: 'Content-Type',
	/** Partial-redaction rules for MCP headers: header name -> visible trailing chars */
	PARTIAL_REDACT: new Map<string, number>([['mcp-session-id', MCP_SESSION_ID_VISIBLE_CHARS]]),

	/** Header names whose values should be redacted in diagnostic logs */
	REDACTED: new Set([
		'authorization',
		'api-key',
		'cookie',
		'mcp-session-id',
		'proxy-authorization',
		'set-cookie',
		'x-auth-token',
		'x-api-key'
	]),

	/** Header carrying the stream-session identity (conversation id, optionally with a model suffix) */
	X_CONVERSATION_ID_HEADER: 'X-Conversation-Id',

	/** Header asking the server to encode a tool's output differently, e.g. read_file returning base64. */
	X_RESP_TYPE_HEADER: 'x-resp-type',

	/** Header carrying the working directory a tool call runs in; the model cannot override it */
	X_TOOL_CWD_HEADER: 'x-tool-cwd'
};

/** `X_RESP_TYPE_HEADER` value that makes read_file return raw bytes as base64 instead of text. */
export const RESP_TYPE_BASE64 = 'base64';
