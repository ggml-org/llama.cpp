<script lang="ts">
	import * as HoverCard from '$lib/components/ui/hover-card';
	import { Button } from '$lib/components/ui/button';
	import { untrack } from 'svelte';
	import { Loader2 } from '@lucide/svelte';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { chatStore, isLoading, isChatStreaming } from '$lib/stores/chat.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { activeConversation, activeMessages } from '$lib/stores/conversations.svelte';
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
			const cached = modelsStore.getModelProps(activeModelId);
			if (!cached) {
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

	let contextUsed = $derived(processingState.processingState?.contextUsed ?? 0);

	// Aggregate reading tokens, output tokens, and average generation speed over
	// every assistant message in the active conversation so the gauge reflects
	// the cumulative LLM work across the whole conversation, not just the most
	// recent step. For agentic messages we use the agentic.llm rollup (covers
	// every tool-loop iteration); for plain messages the top-level timings
	// already describe the full turn. The live processingState is added on top
	// during streaming — chat.sendMessage.onComplete always clears
	// processingState on stream end, so mid-flight the live values for the
	// current running turn never overlap with any message's persisted timings
	// (and the agentic.llm rollup only sums COMPLETED turns anyway).
	let conversationStats = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];
		const live = processingState.processingState;
		const isLive = isLoading() || isChatStreaming();

		let readTokens = 0;
		let outputTokens = 0;
		let outputMs = 0;

		for (const m of messages) {
			if (m.role !== MessageRole.ASSISTANT) continue;
			const timings = m.timings;
			if (!timings) continue;

			const llm = timings.agentic?.llm;
			if (llm?.prompt_n != null || llm?.predicted_n != null || llm?.predicted_ms != null) {
				readTokens += llm?.prompt_n ?? 0;
				outputTokens += llm?.predicted_n ?? 0;
				outputMs += llm?.predicted_ms ?? 0;
			} else {
				readTokens += timings.prompt_n ?? 0;
				outputTokens += timings.predicted_n ?? 0;
				outputMs += timings.predicted_ms ?? 0;
			}
		}

		// Live contribution from the in-flight turn: promptTokens grows during
		// the read phase, outputTokensUsed grows during the gen phase. SSE
		// emits prompt_progress events separately from timings, so during the
		// pure reading phase promptTokens can be 0 even though reading is
		// actively progressing — fall back to promptProgress.processed in
		// that case (it's the canonical server-reported read counter). Max
		// keeps us from double-adding when both are populated.
		if (isLive && live) {
			const livePromptProgress = live.promptProgress?.processed ?? 0;
			const liveRead = Math.max(live.promptTokens ?? 0, livePromptProgress);
			const liveOut = live.outputTokensUsed ?? 0;
			if (liveRead > 0) readTokens += liveRead;
			if (liveOut > 0) outputTokens += liveOut;
			const liveTps = live.tokensPerSecond ?? 0;
			if (liveTps > 0 && liveOut > 0) {
				outputMs += (liveOut / liveTps) * 1000;
			}
		}

		const averageTokensPerSecond =
			outputMs > 0 && outputTokens > 0 ? (outputTokens / outputMs) * 1000 : null;

		return { readTokens, outputTokens, averageTokensPerSecond };
	});

	// Per-message output: do NOT show (replaced by conversationStats.outputTokens).
	// Per-message t/s: do NOT show (replaced by conversationStats.averageTokensPerSecond).
	// Context: shown elsewhere in the hover card; speculative decoding is server-level
	// and worth keeping as a single flag.
	let transientDetails = $derived(
		processingState
			.getTechnicalDetails()
			.filter(
				(detail) =>
					!detail.startsWith('Context:') &&
					!detail.startsWith('Output:') &&
					!detail.includes(STATS_UNITS.TOKENS_PER_SECOND)
			)
	);

	// Tool definitions token count - via llama-server /tokenize (cached in
	// toolsStore per "<model>:<enabledDefList>" hash, no-op when nothing
	// changed). null means the value hasn't been measured yet; 0 means no
	// tools are enabled.
	let enabledToolsTokenCount = $derived(toolsStore.enabledToolsTokenCount);

	// Refresh once on mount, then again whenever the active model switches or
	// the enabled tool set changes. getEnabledToolsForLLM reads $state from
	// every source (built-in list, sandbox toggle, MCP connections, custom
	// JSON, disabled set), so this single read subscribes to all of them.
	$effect(() => {
		const modelId = activeModelId;
		untrack(() => {
			// Single read covers all 4 input sources + disabled-tools set.
			const enabled = toolsStore.getEnabledToolsForLLM();
			void enabled;
			toolsStore.refreshEnabledToolsTokenCount(modelId).catch((err) => {
				console.warn('[ChatFormContextGauge] Failed to refresh tools token count:', err);
			});
		});
	});

	let contextPercent = $derived.by(() => {
		if (contextTotal === null || contextTotal <= 0) return null;
		return Math.round((contextUsed / contextTotal) * 100);
	});

	let contextLabelColor = $derived.by(() => {
		if (contextPercent === null) return 'text-muted-foreground';
		if (contextPercent >= 90) return 'text-red-400';
		if (contextPercent >= 75) return 'text-amber-400';
		return 'text-green-400';
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

	// keep chatStore.activeProcessingState aligned with the active conversation
	$effect(() => {
		const conversation = activeConversation();

		untrack(() => chatStore.setActiveProcessingConversation(conversation?.id ?? null));
	});

	// Populate contextUsed from the start of having a conversation:
	// - during streaming the processing state is updated live from SSE timing events
	// - on a finished conversation the last assistant message owns its prompt /
	//   cache / predicted tokens via its timings, so restore from those
	// - on an empty (fresh) conversation, clear so the gauge doesn't carry over
	//   stale lastKnownState from a previous conversation
	$effect(() => {
		const conversation = activeConversation();
		const messages = activeMessages() as DatabaseMessage[];

		if (!conversation) return;

		// live stream takes precedence over any restoration
		if (isLoading() || isChatStreaming()) return;

		if (messages.length === 0) {
			untrack(() => chatStore.clearProcessingState(conversation.id));
			return;
		}

		untrack(() => chatStore.restoreProcessingStateFromMessages(messages, conversation.id));
	});

	// start the monitor once; chatStore clears activeProcessingState on stream
	// end, and useProcessingState keeps lastKnownState around for after-stream display
	$effect(() => {
		processingState.startMonitoring();
	});
</script>

<HoverCard.Root>
	<HoverCard.Trigger class="flex h-7 w-7 cursor-default items-center justify-center">
		<svg viewBox="0 0 32 32" fill="none" class="h-7 w-7">
			<!-- Background track -->
			<circle cx="16" cy="16" r="11" stroke="currentColor" stroke-opacity="0.1" stroke-width="2" />
			<!-- Progress arc -->
			<circle
				cx="16"
				cy="16"
				r="11"
				class="transition-colors duration-300"
				class:text-green-400={contextPercent !== null && contextPercent < 75}
				class:text-amber-400={contextPercent !== null &&
					contextPercent >= 75 &&
					contextPercent < 90}
				class:text-red-400={contextPercent !== null && contextPercent >= 90}
				class:text-muted-foreground={contextPercent === null}
				stroke="currentColor"
				stroke-width="2"
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
						class:bg-green-500={contextPercent !== null && contextPercent < 75}
						class:bg-amber-500={contextPercent !== null &&
							contextPercent >= 75 &&
							contextPercent < 90}
						class:bg-red-500={contextPercent !== null && contextPercent >= 90}
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

			{#if (enabledToolsTokenCount !== null && enabledToolsTokenCount > 0) || conversationStats.readTokens > 0 || conversationStats.outputTokens > 0 || conversationStats.averageTokensPerSecond !== null || transientDetails.length > 0}
				<div class="mt-1 flex flex-col gap-1 border-t border-border/50 pt-2 text-xs">
					{#if enabledToolsTokenCount !== null && enabledToolsTokenCount > 0}
						<div class="flex items-baseline justify-between">
							<span class="text-muted-foreground">Tool definitions</span>
							<span class="font-mono text-muted-foreground">
								{formatParameters(enabledToolsTokenCount)}
							</span>
						</div>
						<div class="text-[10px] leading-tight text-muted-foreground/70">
							Sent on every turn, cached after the first
						</div>
					{/if}
					{#if conversationStats.readTokens > 0}
						<div class="flex items-baseline justify-between">
							<span class="text-muted-foreground">Reading</span>
							<span class="font-mono text-muted-foreground">
								{conversationStats.readTokens.toLocaleString()} tok
							</span>
						</div>
					{/if}
					{#if conversationStats.outputTokens > 0}
						<div class="flex items-baseline justify-between">
							<span class="text-muted-foreground">Total output</span>
							<span class="font-mono text-muted-foreground">
								{conversationStats.outputTokens.toLocaleString()} tok
							</span>
						</div>
					{/if}
					{#if conversationStats.averageTokensPerSecond !== null}
						<div class="flex items-baseline justify-between">
							<span class="text-muted-foreground">Avg speed</span>
							<span class="font-mono text-muted-foreground">
								{conversationStats.averageTokensPerSecond.toFixed(1)}
								{STATS_UNITS.TOKENS_PER_SECOND}
							</span>
						</div>
					{/if}
					{#each transientDetails as detail (detail)}
						<div class="font-mono text-muted-foreground">{detail}</div>
					{/each}
				</div>
			{/if}
		</div>
	</HoverCard.Content>
</HoverCard.Root>
