import { getContext, setContext } from 'svelte';
import { CONTEXT_KEY_PROCESSING_INFO } from '$lib/constants';

export interface ProcessingInfoContext {
	readonly showProcessingInfo: boolean;
}

const PROCESSING_INFO_KEY = Symbol.for(CONTEXT_KEY_PROCESSING_INFO);

export function setProcessingInfoContext(ctx: ProcessingInfoContext): ProcessingInfoContext {
	return setContext(PROCESSING_INFO_KEY, ctx);
}

export function getProcessingInfoContext(): ProcessingInfoContext {
	return getContext(PROCESSING_INFO_KEY);
}
