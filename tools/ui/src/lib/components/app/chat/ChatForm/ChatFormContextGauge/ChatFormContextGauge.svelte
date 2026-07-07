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

	// Live token stats from the server's processing state, parsed once per
	// update so every consumer derived below reruns in lockstep with each
	// server emission. Null outside of preparing/generating.
	//
	// promptTokens is the combined reading (prompt_n + cache_n) — during preparing
	// it's promptProgress.processed (= total prompt tokens from the server), during
	// generating it falls back to prompt_n since promptProgress disappears after
	// pre_decode. cacheTokens is kept for reference but should not be added to
	// promptTokens — that would double-count since promptTokens already includes
	// the cached portion.
	let liveStats = $derived.by(() => {
		const live = processingState.processingState;
		if (!live || (live.status !== 'preparing' && live.status !== 'generating')) {
			return null;
		}
		const livePromptProgress = live.promptProgress?.processed ?? 0;
		const livePromptTokens = Math.max(live.promptTokens ?? 0, livePromptProgress);
		return {
			promptTokens: livePromptTokens,
			cacheTokens: live.cacheTokens ?? 0,
			outputTokens: live.outputTokensUsed ?? 0,
			tokensPerSecond: live.tokensPerSecond ?? 0
		};
	});

	// Derive context from message data, plus live in-flight tokens when present.
	// During preparing, live.promptTokens grows as the prompt is processed. During
	// generating, the server zeroes prompt_n (pre_decode) so the value lags, but
	// live.promptTokens still tracks via promptProgress. After the turn, the
	// committed assistant message timings are stable and liveStats is null.
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

		const live = liveStats;
		if (live) {
			// promptTokens already includes cache (combined reading from promptProgress)
			const liveTotal = live.promptTokens + live.outputTokens;
			if (liveTotal > 0) used = Math.max(used, liveTotal);
		}

		return used + (enabledToolsTokenCount ?? 0);
	});

	let cumulativeStats = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];

		let read = 0;
		let output = 0;
		let outputMs = 0;
		let hasAgenticFlow = false;

		// Sum the committed assistant message timings only. The "Cumulative
		// (all responses)" rows deliberately reflect completed turns, so live
		// in-flight tokens are not layered on top - that double-counts whenever
		// a new turn starts after one has already committed. Live reactivity is
		// handled by the "Current request (KV cache)" section below, which reads
		// the same liveStats source directly.
		const agenticMessages = messages.filter(
			(m) => m.role === MessageRole.ASSISTANT && m.timings?.agentic?.llm?.predicted_n != null
		);
		hasAgenticFlow = agenticMessages.length > 0;
		if (hasAgenticFlow) {
			// agentic.llm.predicted_n/prompt_n accumulates across the entire flow
			// and is shared on all assistant messages, so count it from the latest
			// agentic message once.
			const lastAgentic = agenticMessages[agenticMessages.length - 1];
			read += lastAgentic.timings.agentic.llm.prompt_n ?? 0;
			output += lastAgentic.timings.agentic.llm.predicted_n ?? 0;
			outputMs += lastAgentic.timings.agentic.llm.predicted_ms ?? 0;
		} else {
			for (const message of messages) {
				if (message.role !== MessageRole.ASSISTANT) continue;
				const timings = message.timings;
				if (!timings) continue;
				read += timings.prompt_n ?? 0;
				read += timings.cache_n ?? 0;
				output += timings.predicted_n ?? 0;
				outputMs += timings.predicted_ms ?? 0;
			}
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

	// Current request's Reading: prompt_n + cache_n for the in-flight request.
	// During preparing, live.promptTokens tracks prompt processing; during
	// generating, the server zeros prompt_n (pre_decode) but liveStats still
	// carries the last known value. Falls back to the last assistant message's
	// timings when the server has no live data.
	let currentRead = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];

		let read = 0;
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i];
			if (msg.role === MessageRole.ASSISTANT && msg.timings) {
				read = (msg.timings.prompt_n ?? 0) + (msg.timings.cache_n ?? 0);
				break;
			}
		}

		// live.promptTokens is already the combined reading (prompt + cache).
		// Do NOT add live.cacheTokens here — promptTokens includes cache_n via
		// promptProgress.processed from the server.
		const live = liveStats;
		if (live && live.promptTokens > 0) {
			read = Math.max(read, live.promptTokens);
		}

		return read;
	});

	// Current request's Output: tokens generated so far in this response.
	let currentOutput = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];

		const live = liveStats;
		if (live && live.outputTokens > 0) return live.outputTokens;

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

					<Collapsible.Content class="flex flex-col gap-4 text-xs pt-4">
						{#if cumulativeStats.read > 0 || cumulativeStats.output > 0}
							<div>
    							<h3
    								class="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/70 mb-2"
    							>
									Across all turns
                                </h3>

								<div class="flex flex-col gap-2">
									{#if cumulativeStats.read > 0}
										<ContextGaugeDetailRow
											label="Prompts sent"
											value={`${cumulativeStats.read.toLocaleString()} tok`}
										/>
									{/if}
									{#if cumulativeStats.output > 0}
										<ContextGaugeDetailRow
											label="Tokens generated"
											value={`${cumulativeStats.output.toLocaleString()} tok`}
										/>
									{/if}
								</div>
							</div>
						{/if}

						{#if currentRead > 0 || currentOutput > 0 || (enabledToolsTokenCount ?? 0) > 0}
							<div>
								<h3
									class="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/70 mb-2"
								>
									This turn · KV cache
								</h3>

								<div class="flex flex-col gap-2">
    								{#if enabledToolsTokenCount ?? 0 > 0}
       									<ContextGaugeDetailRow
      										label="Tool schema"
      										value={`${(enabledToolsTokenCount ?? 0).toLocaleString()} tok`}
      										subtitle="Sent on every turn, cached after the first"
       									/>
    								{/if}

									{#if currentRead > 0}
										<ContextGaugeDetailRow
											label="Prompt"
											value={`${currentRead.toLocaleString()} tok`}
										/>
									{/if}

									{#if currentOutput > 0}
										<ContextGaugeDetailRow
											label="Generated"
											value={`${currentOutput.toLocaleString()} tok`}
										/>
									{/if}

									<div class="pt-1 mt-0.5 border-t border-border/30">
										<div class="flex justify-between">
											<span class="text-muted-foreground">KV cache total</span>
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
