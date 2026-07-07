import { conversationsStore } from '$lib/stores/conversations.svelte';
import { modelsStore, selectedModelContextSize, singleModelName } from '$lib/stores/models.svelte';
import { isRouterMode } from '$lib/stores/server.svelte';
import { DatabaseService } from '$lib/services/database.service';
import { formatParameters } from '$lib/utils/formatters';
import { MessageRole } from '$lib/enums';
import type { ApiProcessingState, ChatMessageTimings, DatabaseMessage } from '$lib/types';

const DEBUG = () => import.meta.env.DEV && import.meta.env.VITE_DEBUG;

export function logResponseStats(timings: ChatMessageTimings | undefined): void {
	if (!DEBUG()) return;

	const assistantMessages = (conversationsStore.activeMessages as DatabaseMessage[]).filter(
		(m) => m.role === MessageRole.ASSISTANT
	);
	const index = assistantMessages.length;
	const prompt = timings?.prompt_n ?? 0;
	const predicted = timings?.predicted_n ?? 0;
	console.log(`[ChatStore] response #${index}: ${prompt} reading / ${predicted} generation`);
}

export function logFlowSummary(): void {
	if (!DEBUG()) return;

	const messages = conversationsStore.activeMessages as DatabaseMessage[];
	const assistantMessages = messages.filter((m) => m.role === MessageRole.ASSISTANT);
	const agenticMessages = assistantMessages.filter(
		(m) => m.timings?.agentic?.llm?.predicted_n != null
	);

	let contextUsed = 0;
	for (const msg of assistantMessages) {
		if (!msg.timings) continue;
		contextUsed += (msg.timings.prompt_n ?? 0) + (msg.timings.predicted_n ?? 0);
	}
	const contextTotal = modelsStore.getModelContextSize(singleModelName() ?? '');
	const contextPercent =
		contextTotal && contextTotal > 0 ? Math.round((contextUsed / contextTotal) * 100) : null;

	console.group(
		`[ChatStore] --- TOKEN AUDIT --- Context: ${formatParameters(contextUsed)} / ${contextTotal !== null ? formatParameters(contextTotal) : '—'} (${contextPercent !== null ? contextPercent + '%' : '?'})`
	);

	let totalRead = 0;
	let totalOutput = 0;
	let agg: { prompt_n?: number; predicted_n?: number } | undefined;

	if (agenticMessages.length > 0) {
		const lastAgentic = agenticMessages[agenticMessages.length - 1];
		agg = lastAgentic.timings?.agentic?.llm;
		totalRead = agg?.prompt_n ?? 0;
		totalOutput = agg?.predicted_n ?? 0;

		assistantMessages.forEach((msg, idx) => {
			const t = msg.timings;
			const modelLabel = msg.model ? ` [${msg.model}]` : '';
			if (t?.agentic?.llm?.predicted_n != null) {
				console.log(`  #${idx + 1}${modelLabel}`);
				console.log(`    Fresh:        ${t.agentic.llm.prompt_n ?? 0}`);
				console.log(`    Generated:    ${t.agentic.llm.predicted_n ?? 0}`);
			} else {
				const fresh = t?.prompt_n ?? 0;
				const cached = t?.cache_n ?? 0;
				const totalInput = fresh + cached;
				const generated = t?.predicted_n ?? 0;
				console.log(`  #${idx + 1}${modelLabel}`);
				console.log(`    Fresh:        ${fresh}`);
				console.log(`    Cached:       ${cached}`);
				console.log(`    Total input:  ${totalInput}`);
				console.log(`    Generated:    ${generated}`);
			}
		});
	} else {
		assistantMessages.forEach((msg, idx) => {
			const t = msg.timings;
			const fresh = t?.prompt_n ?? 0;
			const cached = t?.cache_n ?? 0;
			const totalInput = fresh + cached;
			const generated = t?.predicted_n ?? 0;
			totalRead += totalInput;
			totalOutput += generated;
			const modelLabel = msg.model ? ` [${msg.model}]` : '';
			console.log(`  #${idx + 1}${modelLabel}`);
			console.log(`    Fresh:        ${fresh}`);
			console.log(`    Cached:       ${cached}`);
			console.log(`    Total input:  ${totalInput}`);
			console.log(`    Generated:    ${generated}`);
		});
	}

	const lastMsg = assistantMessages[assistantMessages.length - 1];
	const currentRead =
		agenticMessages.length > 0
			? (agg?.prompt_n ?? 0)
			: lastMsg?.timings
				? (lastMsg.timings.prompt_n ?? 0) + (lastMsg.timings.cache_n ?? 0)
				: 0;
	const currentOutput =
		agenticMessages.length > 0 ? (agg?.predicted_n ?? 0) : (lastMsg?.timings?.predicted_n ?? 0);

	console.log('%cCumulative (all responses):', 'font-weight: bold');
	console.log(`  Total input: ${formatParameters(totalRead)} tok`);
	console.log(`  Generated:   ${formatParameters(totalOutput)} tok`);

	console.log('%cCurrent (KV cache):', 'font-weight: bold');
	console.log(`  Total input: ${formatParameters(currentRead)} tok`);
	console.log(`  Generated:   ${formatParameters(currentOutput)} tok`);
	console.log(`  ─────────────────────`);
	console.log(`  KV total:    ${formatParameters(currentRead + currentOutput)} tok`);

	console.groupEnd();
}

export async function debugTokenUsage(
	activeProcessingState: ApiProcessingState | null,
	convIdOrUndefined?: string
): Promise<void> {
	if (!DEBUG()) return;

	let convId = convIdOrUndefined;
	if (!convId) {
		const active = conversationsStore.activeConversation;
		if (!active) {
			console.log('[Debug] No active conversation');
			return;
		}
		convId = active.id;
	}

	const messages = await DatabaseService.getConversationMessages(convId as string);
	const assistantMessages = messages.filter((m) => m.role === MessageRole.ASSISTANT);
	const agenticMessages = assistantMessages.filter(
		(m) => m.timings?.agentic?.llm?.predicted_n != null
	);

	let contextUsed = 0;
	for (const msg of assistantMessages) {
		if (!msg.timings) continue;
		contextUsed += (msg.timings.prompt_n ?? 0) + (msg.timings.predicted_n ?? 0);
	}
	const contextTotal = modelsStore.getModelContextSize(singleModelName() ?? '');
	const contextPercent =
		contextTotal && contextTotal > 0 ? Math.round((contextUsed / contextTotal) * 100) : null;

	console.group(
		`[Debug] Token audit: ${convId} (${assistantMessages.length} responses) Context: ${formatParameters(contextUsed)} / ${contextTotal !== null ? formatParameters(contextTotal) : '—'} (${contextPercent !== null ? contextPercent + '%' : '?'})`
	);

	console.log('%cPer-message token usage:', 'font-weight: bold');
	let totalRead = 0;
	let totalOutput = 0;
	let agg: { prompt_n?: number; predicted_n?: number } | undefined;

	if (agenticMessages.length > 0) {
		const lastAgentic = agenticMessages[agenticMessages.length - 1];
		agg = lastAgentic.timings?.agentic?.llm;
		totalRead = agg?.prompt_n ?? 0;
		totalOutput = agg?.predicted_n ?? 0;

		assistantMessages.forEach((msg, idx) => {
			const t = msg.timings;
			const modelLabel = msg.model ? ` [${msg.model}]` : '';
			if (t?.agentic?.llm?.predicted_n != null) {
				console.log(
					`  #${idx + 1}: agentic (llm.predicted_n=${t.agentic.llm.predicted_n}, llm.prompt_n=${t.agentic.llm.prompt_n}) ${modelLabel}`
				);
			} else {
				const read = (t?.prompt_n ?? 0) + (t?.cache_n ?? 0);
				const output = t?.predicted_n ?? 0;
				console.log(
					`  #${idx + 1}: prompt=${t?.prompt_n ?? 0} cache=${t?.cache_n ?? 0} reading=${read} | output=${output} ${modelLabel}`
				);
			}
		});
	} else {
		assistantMessages.forEach((msg, idx) => {
			const t = msg.timings;
			const read = (t?.prompt_n ?? 0) + (t?.cache_n ?? 0);
			const output = t?.predicted_n ?? 0;
			totalRead += read;
			totalOutput += output;

			const modelLabel = msg.model ? ` [${msg.model}]` : '';
			console.log(
				`  #${idx + 1}: prompt=${t?.prompt_n ?? 0} cache=${t?.cache_n ?? 0} reading=${read} | output=${output} ${modelLabel}`
			);
		});
	}

	const lastMsg = assistantMessages[assistantMessages.length - 1];
	const currentRead =
		agenticMessages.length > 0
			? (agg?.prompt_n ?? 0)
			: lastMsg?.timings
				? (lastMsg.timings.prompt_n ?? 0) + (lastMsg.timings.cache_n ?? 0)
				: 0;
	const currentOutput =
		agenticMessages.length > 0 ? (agg?.predicted_n ?? 0) : (lastMsg?.timings?.predicted_n ?? 0);

	console.log('%cCumulative (all responses):', 'font-weight: bold');
	console.log(`  Reading:   ${formatParameters(totalRead)} tok`);
	console.log(`  Output:    ${formatParameters(totalOutput)} tok`);

	console.log('%cCurrent (KV cache):', 'font-weight: bold');
	console.log(`  Reading:   ${formatParameters(currentRead)} tok`);
	console.log(`  Output:    ${formatParameters(currentOutput)} tok`);

	const serverContext = activeProcessingState?.contextUsed ?? 0;
	console.log('%cLive gauge values:', 'font-weight: bold');
	console.log(`  contextUsed (server):        ${formatParameters(serverContext)} tok`);
	console.log(
		`  contextTotal (model):        ${formatParameters(
			isRouterMode()
				? (selectedModelContextSize() ?? 'unknown')
				: singleModelName()
					? modelsStore.getModelContextSize(singleModelName()!)
					: 'unknown'
		)} tok`
	);

	if (serverContext > 0 && currentRead > 0) {
		const diff = serverContext - (currentRead + currentOutput);
		if (diff !== 0) {
			console.log(
				`  Server vs calculated delta: ${formatParameters(diff)} tok`,
				diff > 0 ? '(system prompt + overhead)' : ''
			);
		} else {
			console.log('  ✓ Server and calculated context match');
		}
	}

	console.groupEnd();
}
