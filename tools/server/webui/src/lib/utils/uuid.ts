export function uuid(): string {
	return globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).substring(2);
}
