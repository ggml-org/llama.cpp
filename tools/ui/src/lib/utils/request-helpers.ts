/**
 * HTTP request inspection utilities for diagnostic logging.
 * These helpers extract metadata from fetch-style request arguments
 * without exposing sensitive payload data.
 */

export interface RequestBodySummary {
	kind: string;
	size?: number;
}

export function getRequestUrl(input: RequestInfo | URL): string {
	if (typeof input === 'string') {
		return input;
	}

	if (input instanceof URL) {
		return input.href;
	}

	return input.url;
}

export function getRequestMethod(
	input: RequestInfo | URL,
	init?: RequestInit,
	baseInit?: RequestInit
): string {
	if (init?.method) {
		return init.method;
	}

	if (typeof Request !== 'undefined' && input instanceof Request) {
		return input.method;
	}

	return baseInit?.method ?? 'GET';
}

export function getRequestBody(
	input: RequestInfo | URL,
	init?: RequestInit
): BodyInit | null | undefined {
	if (init?.body !== undefined) {
		return init.body;
	}

	if (typeof Request !== 'undefined' && input instanceof Request) {
		return input.body;
	}

	return undefined;
}

export function summarizeRequestBody(body: BodyInit | null | undefined): RequestBodySummary {
	if (body == null) {
		return { kind: 'empty' };
	}

	if (typeof body === 'string') {
		return { kind: 'string', size: body.length };
	}

	if (body instanceof Blob) {
		return { kind: 'blob', size: body.size };
	}

	if (body instanceof URLSearchParams) {
		return { kind: 'urlsearchparams', size: body.toString().length };
	}

	if (body instanceof FormData) {
		return { kind: 'formdata' };
	}

	if (body instanceof ArrayBuffer) {
		return { kind: 'arraybuffer', size: body.byteLength };
	}

	if (ArrayBuffer.isView(body)) {
		return { kind: body.constructor.name, size: body.byteLength };
	}

	return { kind: typeof body };
}

export function formatDiagnosticErrorMessage(error: unknown): string {
	const message = error instanceof Error ? error.message : String(error);

	return message.includes('Failed to fetch') ? `${message} (check CORS?)` : message;
}

export function extractJsonRpcMethods(body: BodyInit | null | undefined): string[] | undefined {
	if (typeof body !== 'string') {
		return undefined;
	}

	try {
		const parsed = JSON.parse(body);
		const messages = Array.isArray(parsed) ? parsed : [parsed];
		const methods = messages
			.map((message: Record<string, unknown>) =>
				typeof message?.method === 'string' ? (message.method as string) : undefined
			)
			.filter((method: string | undefined): method is string => Boolean(method));

		return methods.length > 0 ? methods : undefined;
	} catch {
		return undefined;
	}
}
