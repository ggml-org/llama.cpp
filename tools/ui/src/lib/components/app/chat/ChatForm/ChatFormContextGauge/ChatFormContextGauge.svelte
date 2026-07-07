<script lang="ts">
	import * as Collapsible from '$lib/components/ui/collapsible';
	import * as HoverCard from '$lib/components/ui/hover-card';
	import { Button } from '$lib/components/ui/button';
	import { untrack } from 'svelte';
	import { ChevronDown, Loader2 } from '@lucide/svelte';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { chatStore, isLoading, isChatStreaming } from '$lib/stores/chat.svelte';
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
	// Three live fields matter:
	//   freshTokens — NEW tokens added this turn (= timing.prompt_n =
	//     n_prompt_tokens_processed from server-context.cpp). Excludes cache
	//     hits so the gauge can sum it across turns without double-counting.
	//   promptTokens — full prompt input this turn = fresh + cache hit prefix.
	//     Used by "this turn" breakdown rows.
	//   cacheTokens — matched prefix tokens from the in-memory KV cache.
	let liveStats = $derived.by(() => {
		const live = processingState.processingState;
		if (!live || (live.status !== 'preparing' && live.status !== 'generating')) {
			return null;
		}
		const livePromptTokens = live.promptTokens ?? 0;
		const liveCacheTokens = live.cacheTokens ?? 0;
		return {
			freshTokens: livePromptTokens,
			promptTokens: livePromptTokens + liveCacheTokens,
			cacheTokens: liveCacheTokens,
			outputTokens: live.outputTokensUsed ?? 0,
			tokensPerSecond: live.tokensPerSecond ?? 0
		};
	});

	// Derive context from message data, plus live in-flight tokens when present.
	//
	// KV cache in memory = sum across all committed assistant turns of
	// (timings.prompt_n + timings.predicted_n). Cache hits are excluded because
	// they reference tokens already counted by a prior turn's prompt_n.
	//
	// During streaming of a new turn, the previous cumulative total still
	// represents the slot's KV. The live "fresh" addition (= this turn's new
	// prompt_n) and the live output are added on top — that's the running
	// delta between committed state and the in-flight request.
	//
	// Edge case: there's a brief window between commitMessageAtIndex(...) (the
	// assistant timings land in activeMessages) and cleanupStreamingState()
	// (processingState clears). During that window the same turn's data lives
	// in BOTH the committed array and the live state, so adding live on top of
	// sum would double-count. Guard against that by skipping live additions
	// whenever the last entry in activeMessages is already a committed
	// assistant — the live state at that moment is just the same delta.
	let contextUsed = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];

		let used = 0;
		for (const msg of messages) {
			if (msg.role !== MessageRole.ASSISTANT || !msg.timings) continue;
			used += (msg.timings.prompt_n ?? 0) + (msg.timings.predicted_n ?? 0);
		}

		const live = liveStats;
		const lastMsg = messages[messages.length - 1];
		const liveTurnAlreadyCommitted =
			!!lastMsg && lastMsg.role === MessageRole.ASSISTANT && !!lastMsg.timings;
		if (live && !liveTurnAlreadyCommitted) {
			used += live.freshTokens + live.outputTokens;
		}

		return used;
	});

	let cumulativeStats = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];

		let read = 0;
		let output = 0;
		let outputMs = 0;
		let cacheTotal = 0;
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
			// agentic message once. agentic flows don't surface per-turn cache_n,
			// so cacheTotal stays 0.
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
				cacheTotal += timings.cache_n ?? 0;
				output += timings.predicted_n ?? 0;
				outputMs += timings.predicted_ms ?? 0;
			}
		}

		const averageTokensPerSecond = outputMs > 0 && output > 0 ? (output / outputMs) * 1000 : null;

		return { read, output, cacheTotal, averageTokensPerSecond };
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

	// Current request's fresh (newly processed) tokens: this turn's prompt_n.
	// live.freshTokens is the server's n_prompt_tokens_processed - cache hits
	// already excluded, so it matches the committed semantics exactly.
	let currentFresh = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];

		let fresh = 0;
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i];
			if (msg.role === MessageRole.ASSISTANT && msg.timings) {
				fresh = msg.timings.prompt_n ?? 0;
				break;
			}
		}

		const live = liveStats;
		if (live) {
			fresh = Math.max(fresh, live.freshTokens);
		}

		return fresh;
	});

	// Current request's cached tokens: cache_n for the last assistant message —
	// those that matched the in-memory KV cache and were not re-processed.
	let currentCache = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];

		let cached = 0;
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i];
			if (msg.role === MessageRole.ASSISTANT && msg.timings) {
				cached = msg.timings.cache_n ?? 0;
				break;
			}
		}

		const live = liveStats;
		if (live && live.promptTokens > 0) {
			cached = Math.max(cached, live.cacheTokens);
		}

		return cached;
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

	// KV total = current request's Reading + Output. The model's prompt_n + cache_n
	// already includes tool definitions rendered into the prompt by the chat template.
	let kvTotal = $derived(currentRead + currentOutput);

	let detailsOpen = $state(false);

	let hasDetails = $derived(
		cumulativeStats.read > 0 ||
			cumulativeStats.output > 0 ||
			currentRead > 0 ||
			currentOutput > 0 ||
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
											subtitle={cumulativeStats.cacheTotal > 0
												? `${cumulativeStats.cacheTotal.toLocaleString()} of these were cached (KV hit)`
												: undefined}
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

						{#if currentRead > 0 || currentOutput > 0}
							<div>
								<h3
									class="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/70 mb-2"
								>
									This turn · KV cache
								</h3>

								<div class="flex flex-col gap-2">
									{#if currentRead > 0}
										<ContextGaugeDetailRow
											label="Prompt"
											value={`${currentRead.toLocaleString()} tok`}
											subtitle={currentCache > 0
												? `${currentFresh.toLocaleString()} fresh + ${currentCache.toLocaleString()} cached`
												: undefined}
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
