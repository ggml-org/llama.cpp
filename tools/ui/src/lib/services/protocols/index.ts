/**
 * Protocol adapter registry.
 *
 * Resolves the wire mapping for a backend by its protocol. Unknown or
 * not-yet-loaded backends fall back to the OpenAI-compatible adapter, which is
 * what the local llama-server speaks.
 */

import { openaiAdapter } from './openai';
import type { ChatProtocolAdapter } from './types';
import { BackendProtocol } from '$lib/constants';
import type { Backend } from '$lib/types';

const ADAPTERS: Record<BackendProtocol, ChatProtocolAdapter> = {
	[BackendProtocol.COMPAT]: openaiAdapter,
	[BackendProtocol.OPENAI]: openaiAdapter
};

export function getProtocolAdapter(backend?: Backend): ChatProtocolAdapter {
	if (!backend) return openaiAdapter;

	return ADAPTERS[backend.protocol] ?? openaiAdapter;
}

export type { ChatProtocolAdapter, ChatStreamEvent, ChatStreamReader } from './types';
