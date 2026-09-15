/**
 * Protocol adapter registry.
 *
 * Resolves the wire mapping for a backend by its protocol. Unknown or
 * not-yet-loaded backends fall back to the OpenAI-compatible adapter, which is
 * what the local llama-server speaks.
 */

import { anthropicAdapter } from './anthropic';
import { openaiAdapter } from './openai';
import type { ChatProtocolAdapter } from './types';
import type { Backend, BackendProtocol } from '$lib/types';

const ADAPTERS: Record<BackendProtocol, ChatProtocolAdapter> = {
	anthropic: anthropicAdapter,
	'llama.cpp': openaiAdapter,
	openai: openaiAdapter
};

export function getProtocolAdapter(backend?: Backend): ChatProtocolAdapter {
	if (!backend) return openaiAdapter;

	return ADAPTERS[backend.protocol] ?? openaiAdapter;
}

export type { ChatProtocolAdapter, ChatStreamEvent, ChatStreamReader } from './types';
