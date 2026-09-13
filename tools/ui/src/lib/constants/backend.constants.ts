import type { BackendProtocol } from '$lib/types';

/** Id of the built-in backend that points at the server serving this UI. */
export const LOCAL_BACKEND_ID = 'local';

/** Protocols a configured backend can speak, in display order. */
export const BACKEND_PROTOCOLS: readonly BackendProtocol[] = ['llama.cpp', 'openai', 'anthropic'];

/** Prefix for generated ids of user-added backends. */
export const BACKEND_ID_PREFIX = 'backend';
