<script lang="ts">
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as HoverCard from '$lib/components/ui/hover-card';
	import { Button } from '$lib/components/ui/button';
	import { untrack } from 'svelte';
	import { ChevronDown, Loader2 } from '@lucide/svelte';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { chatStore, isLoading, isChatStreaming } from '$lib/stores/chat.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { activeConversation, activeMessages } from '$lib/stores/conversations.svelte';
	import ContextGaugeDetailRow from './ContextGaugeDetailRow.svelte';
	import {
		modelsStore,
		modelOptions,
		selectedModelId,
		singleModelName
	} from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { MessageRole } from '$lib/enums';
	import { STATS_UNITS } from '$lib/constants';
	import { formatParameters } from '$lib/utils/formatters';

	const processingState = useProcessingState();

	// Resolve the active model the gauge reports context for. Mirrors the resolver
	// in useChatScreenActiveModel: explicit user selection wins, otherwise the
	// active conversation's last assistant model, otherwise the single loaded
	// model in MODEL mode (null otherwise).
	let activeModelId = $derived.by(() => {
		if (!isRouterMode()) {
			return singleModelName();
		}

		const selectedId = selectedModelId();
		if (selectedId) {
			const model = modelOptions().find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		const conversationModel = chatStore.getConversationModel(activeMessages() as DatabaseMessage[]);
		return conversationModel;
	});

	let isActiveModelLoaded = $derived(
		activeModelId !== null && modelsStore.isModelLoaded(activeModelId)
	);

	let isActiveModelLoading = $derived(
		activeModelId !== null && modelsStore.isModelOperationInProgress(activeModelId)
	);

	// Auto-fetch /props?model=<id> when the active model is loaded but props aren't
	// yet in the cache, so n_ctx becomes available without sending a chat request.
	$effect(() => {
		if (activeModelId && isActiveModelLoaded) {
			const cachedProps = modelsStore.getModelProps(activeModelId);
			if (!cachedProps) {
				void modelsStore.fetchModelProps(activeModelId);
			}
		}
	});

	let contextTotal = $derived.by(() => {
		void modelsStore.propsCacheVersion;
		if (activeModelId) {
			return modelsStore.getModelContextSize(activeModelId);
		}
		return null;
	});

	// Derive context from message data (same approach as logFlowSummary).
	// Do NOT use processingState.processingState?.contextUsed — it is cleared during
	// streaming and only restored after the response finishes, making it always one
	// message behind.
	let contextUsed = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];
		const agenticMessages = messages.filter(
			(m) => m.role === MessageRole.ASSISTANT && m.timings?.agentic?.llm?.predicted_n != null
		);

		let used = 0;
		if (agenticMessages.length > 0) {
			const lastAgentic = agenticMessages[agenticMessages.length - 1];
			const agg = lastAgentic.timings?.agentic?.llm;
			used = (agg?.prompt_n ?? 0) + (agg?.predicted_n ?? 0);
		} else if (messages.length > 0) {
			for (let i = messages.length - 1; i >= 0; i--) {
				const msg = messages[i];
				if (msg.role === MessageRole.ASSISTANT && msg.timings) {
					used =
						(msg.timings.prompt_n ?? 0) +
						(msg.timings.cache_n ?? 0) +
						(msg.timings.predicted_n ?? 0);
					break;
				}
			}
		}
		return used + (enabledToolsTokenCount ?? 0);
	});

	// Track the last known prompt_n and cache_n from the preparing phase.
	// The server resets prompt_n to 0 during generation (pre_decode), so we
	// preserve the final counts from when promptProgress disappears.
	// IMPORTANT: prompt_n and cache_n are separate fields — do NOT add them
	// together here. The "reading" (prompt + cache) is computed in currentRead.
	let lastKnownPromptTokens = $state(0);
	let lastKnownCacheTokens = $state(0);

	$effect(() => {
		const live = processingState.processingState;

		// Reset at the start of a fresh preparing phase (new request).
		if (live?.status === 'preparing') {
			const pp = live.promptProgress;
			if (pp && pp.total > 0 && pp.processed === 0) {
				lastKnownPromptTokens = 0;
				lastKnownCacheTokens = 0;
			}
		}

		if (live?.promptProgress) {
			// Update while prompt processing is ongoing.
			lastKnownPromptTokens = live.promptTokens ?? 0;
			lastKnownCacheTokens = live.cacheTokens ?? 0;
		} else if (live?.status === 'generating' && lastKnownPromptTokens === 0) {
			// Prompt processing just finished — lock in whatever we have before
			// the server zeroes prompt_n during generation.
			lastKnownPromptTokens = live.promptTokens ?? 0;
			lastKnownCacheTokens = live.cacheTokens ?? 0;
		}
	});

	let cumulativeStats = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];

		let read = 0;
		let output = 0;
		let outputMs = 0;
		let hasAgenticFlow = false;

		// Find agentic messages. agentic.llm.predicted_n/prompt_n is accumulated across
		// the entire flow and shared on ALL assistant messages, so we count it only once
		// (from the last agentic message which has the complete accumulated totals).
		const agenticMessages = messages.filter(
			(m) => m.role === MessageRole.ASSISTANT && m.timings?.agentic?.llm?.predicted_n != null
		);
		if (agenticMessages.length > 0) {
			hasAgenticFlow = true;
			const lastAgentic = agenticMessages[agenticMessages.length - 1];
			read += lastAgentic.timings.agentic.llm.prompt_n ?? 0;
			read += lastAgentic.timings.agentic.llm.prompt_ms != null
				? 0 // agentic llm doesn't track cache_n separately; prompt_n is the full prompt
				: 0;
			output += lastAgentic.timings.agentic.llm.predicted_n ?? 0;
			outputMs += lastAgentic.timings.agentic.llm.predicted_ms ?? 0;
		}

		// Non-agentic assistant messages: use per-message timings directly.
		for (const message of messages) {
			if (message.role !== MessageRole.ASSISTANT) continue;
			const timings = message.timings;
			if (!timings) continue;
			if (hasAgenticFlow && timings.agentic?.llm?.predicted_n != null) continue;
			read += timings.prompt_n ?? 0;
			read += timings.cache_n ?? 0;
			output += timings.predicted_n ?? 0;
			outputMs += timings.predicted_ms ?? 0;
		}

		const averageTokensPerSecond = outputMs > 0 && output > 0 ? (output / outputMs) * 1000 : null;

		return { read, output, averageTokensPerSecond };
	});

	const TRANSIENT_DETAILS_EXCLUDED_PREFIXES = ['Context:', 'Output:'];

	let transientDetails = $derived(
		processingState.getTechnicalDetails().filter((technicalDetail) => {
			if (
				TRANSIENT_DETAILS_EXCLUDED_PREFIXES.some((prefix) => technicalDetail.startsWith(prefix))
			) {
				return false;
			}
			return !technicalDetail.includes(STATS_UNITS.TOKENS_PER_SECOND);
		})
	);

	let enabledToolsTokenCount = $derived(toolsStore.enabledToolsTokenCount);

	$effect(() => {
		const modelId = activeModelId;
		const enabledToolsTokenCount = (toolsStore as any)._enabledToolsTokenCount;
		void enabledToolsTokenCount;
		toolsStore.refreshEnabledToolsTokenCount(modelId).catch((err) => {
			console.warn('[ChatFormContextGauge] Failed to refresh tools token count:', err);
		});
	});

	let contextPercent = $derived.by(() => {
		if (contextTotal === null || contextTotal <= 0) return null;
		return Math.round((contextUsed / contextTotal) * 100);
	});

	// Current request's Reading: prompt_n + cache_n for the active request.
	// Falls back to lastKnown values during generation (server zeroes prompt_n).
	let currentRead = $derived.by(() => {
		if (lastKnownPromptTokens > 0) return (lastKnownPromptTokens ?? 0) + (lastKnownCacheTokens ?? 0);
		const messages = activeMessages() as DatabaseMessage[];
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i];
			if (msg.role === MessageRole.ASSISTANT && msg.timings) {
				return (msg.timings.prompt_n ?? 0) + (msg.timings.cache_n ?? 0);
			}
		}
		return 0;
	});

	// Current request's Output: what's being generated right now.
	let currentOutput = $derived.by(() => {
		const live = processingState.processingState;
		if (live && (live.status === 'preparing' || live.status === 'generating')) {
			const liveOutputTokens = live.outputTokensUsed ?? 0;
			if (liveOutputTokens > 0) return liveOutputTokens;
		}
		const messages = activeMessages() as DatabaseMessage[];
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i];
			if (msg.role === MessageRole.ASSISTANT && msg.timings) {
				return msg.timings.predicted_n ?? 0;
			}
		}
		return 0;
	});

	// KV total = current request's Reading + Output + Tool definitions.
	let kvTotal = $derived(currentRead + currentOutput + (enabledToolsTokenCount ?? 0));

	let detailsOpen = $state(false);

	let hasDetails = $derived(
		cumulativeStats.read > 0 ||
			cumulativeStats.output > 0 ||
			currentRead > 0 ||
			currentOutput > 0 ||
			(enabledToolsTokenCount ?? 0) > 0 ||
			cumulativeStats.averageTokensPerSecond !== null ||
			transientDetails.length > 0
	);

	// Auto-expand details during streaming so live token counts are visible.
	$effect(() => {
		const live = processingState.processingState;
		detailsOpen = !!(live && (live.status === 'preparing' || live.status === 'generating'));
	});

	let contextLabelColor = $derived.by(() => {
		if (contextPercent === null) return 'text-muted-foreground';
		if (contextPercent >= 95) return 'text-red-400';
		if (contextPercent >= 80) return 'text-amber-400';
		return 'text-muted-foreground';
	});

	async function handleLoadModel() {
		if (!activeModelId || isActiveModelLoading) return;

		try {
			await modelsStore.loadModel(activeModelId);
		} catch {
			// toast already surfaced by modelsStore.loadModel
		}
	}

	const CIRCUMFERENCE = 2 * Math.PI * 11;

	$effect(() => {
		const currentConversation = activeConversation();

		untrack(() => chatStore.setActiveProcessingConversation(currentConversation?.id ?? null));
	});

	$effect(() => {
		const currentConversation = activeConversation();
		const conversationMessages = activeMessages() as DatabaseMessage[];

		if (!currentConversation) return;

		if (isLoading() || isChatStreaming()) return;

		if (conversationMessages.length === 0) {
			untrack(() => chatStore.clearProcessingState(currentConversation.id));
			return;
		}

		untrack(() =>
			chatStore.restoreProcessingStateFromMessages(conversationMessages, currentConversation.id)
		);
	});

	$effect(() => {
		processingState.startMonitoring();
	});
</script>

<HoverCard.Root>
	<HoverCard.Trigger class="flex h-5 w-5 cursor-default items-center justify-center">
		<svg viewBox="0 0 32 32" fill="none" class="h-5 w-5">
			<!-- Background track -->
			<circle cx="16" cy="16" r="11" stroke="currentColor" stroke-opacity="0.1" stroke-width="3" />
			<!-- Progress arc -->
			<circle
				cx="16"
				cy="16"
				r="11"
				class="transition-colors duration-300"
				class:text-foreground={contextPercent !== null && contextPercent < 80}
				class:text-amber-400={contextPercent !== null &&
					contextPercent >= 80 &&
					contextPercent < 95}
				class:text-red-400={contextPercent !== null && contextPercent >= 95}
				class:text-muted-foreground={contextPercent === null}
				stroke="currentColor"
				stroke-width="3"
				stroke-linecap="round"
				stroke-dasharray={CIRCUMFERENCE}
				stroke-dashoffset={contextPercent !== null
					? CIRCUMFERENCE * (1 - contextPercent / 100)
					: CIRCUMFERENCE}
				transform="rotate(-90 16 16)"
			/>
		</svg>
	</HoverCard.Trigger>

	<HoverCard.Content
		side="bottom"
		class="z-50 w-64 rounded-lg border border-border/50 bg-popover p-3 text-popover-foreground shadow-lg"
	>
		<div class="flex flex-col gap-2">
			<div class="flex items-center gap-2">
				<span class="font-medium">Context</span>
				<span class="text-muted-foreground">·</span>
				<span class="font-mono text-muted-foreground">
					{formatParameters(contextUsed)}
					/ {contextTotal !== null ? formatParameters(contextTotal) : '—'}
				</span>
			</div>

			{#if activeModelId !== null && !isActiveModelLoaded && !isActiveModelLoading}
				<div
					class="flex flex-col gap-2 border-t border-border/50 pt-2 text-xs text-muted-foreground"
				>
					<span> Context size is only available once the model is loaded. </span>
					<Button size="sm" variant="secondary" class="self-start" onclick={handleLoadModel}>
						Load model
					</Button>
				</div>
			{:else if isActiveModelLoading}
				<div
					class="flex items-center gap-2 border-t border-border/50 pt-2 text-xs text-muted-foreground"
				>
					<Loader2 class="h-3.5 w-3.5 animate-spin" />
					<span>Loading model…</span>
				</div>
			{:else if contextTotal !== null && contextTotal > 0}
				<div class="h-1.5 w-full overflow-hidden rounded-full bg-muted">
					<div
						class="h-full rounded-full transition-all duration-300"
						class:bg-green-500={contextPercent !== null && contextPercent < 80}
						class:bg-amber-500={contextPercent !== null &&
							contextPercent >= 80 &&
							contextPercent < 95}
						class:bg-red-500={contextPercent !== null && contextPercent >= 95}
						style="width: {contextPercent}%"
					></div>
				</div>

				<div class="flex justify-between text-xs text-muted-foreground">
					<span>
						<span class={contextLabelColor}>{contextPercent}%</span> used
					</span>
					<span>
						{formatParameters(contextTotal - contextUsed)} remaining
					</span>
				</div>
			{:else}
				<div class="text-xs text-muted-foreground">No context info available</div>
			{/if}

			{#if hasDetails}
				<Collapsible.Root bind:open={detailsOpen} class="mt-3 border-t border-border/50 pt-4">
					<Collapsible.Trigger
						class="flex w-full cursor-pointer items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
					>
						<span>Token usage details</span>

						<ChevronDown
							class={'ml-auto h-3 w-3 transition-transform' + (detailsOpen ? ' rotate-180' : '')}
						/>
					</Collapsible.Trigger>

					<Collapsible.Content class="flex flex-col gap-3 text-xs pt-4">
						{#if cumulativeStats.read > 0 || cumulativeStats.output > 0}
							<div>
								<div class="text-[10px] font-medium uppercase tracking-wide text-muted-foreground/70 mb-1">
									Cumulative (all responses)
								</div>
								<div class="flex flex-col gap-0.5">
									{#if cumulativeStats.read > 0}
										<ContextGaugeDetailRow
											label="Reading"
											value={`${cumulativeStats.read.toLocaleString()} tok`}
										/>
									{/if}
									{#if cumulativeStats.output > 0}
										<ContextGaugeDetailRow
											label="Output"
											value={`${cumulativeStats.output.toLocaleString()} tok`}
										/>
									{/if}
								</div>
							</div>
						{/if}

						{#if currentRead > 0 || currentOutput > 0 || (enabledToolsTokenCount ?? 0) > 0}
							<div>
								<div class="text-[10px] font-medium uppercase tracking-wide text-muted-foreground/70 mb-1">
									Current request (KV cache)
								</div>
								<div class="flex flex-col gap-0.5">
									{#if currentRead > 0}
										<ContextGaugeDetailRow
											label="Reading"
											value={`${currentRead.toLocaleString()} tok`}
										/>
									{/if}
									{#if enabledToolsTokenCount ?? 0 > 0}
										<ContextGaugeDetailRow
											label="Tool definitions"
											value={`${(enabledToolsTokenCount ?? 0).toLocaleString()} tok`}
											subtitle="Sent on every turn, cached after the first"
										/>
									{/if}
									{#if currentOutput > 0}
										<ContextGaugeDetailRow
											label="Output"
											value={`${currentOutput.toLocaleString()} tok`}
										/>
									{/if}
									<div class="pt-1 mt-0.5 border-t border-border/30">
										<div class="flex justify-between">
											<span class="text-muted-foreground">KV total</span>
											<span class="font-mono font-medium">{kvTotal.toLocaleString()} tok</span>
										</div>
									</div>
								</div>
							</div>
						{/if}

						{#if cumulativeStats.averageTokensPerSecond !== null}
							<div class="pt-1.5 mt-1 border-t border-border/30">
								<ContextGaugeDetailRow
									label="Avg speed"
									value={`${cumulativeStats.averageTokensPerSecond.toFixed(1)}${STATS_UNITS.TOKENS_PER_SECOND}`}
								/>
							</div>
						{/if}
						{#each transientDetails as detail (detail)}
							<div class="font-mono text-muted-foreground">{detail}</div>
						{/each}
					</Collapsible.Content>
				</Collapsible.Root>
			{/if}
		</div>
	</HoverCard.Content>
</HoverCard.Root>
