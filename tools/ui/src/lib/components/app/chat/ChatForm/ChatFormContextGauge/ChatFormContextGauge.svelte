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

	// Two scopes for the token-usage breakdown so the rows line up with the
	// context bar AND also reveal how much LLM compute the full run spent:
	//   - currentStats: the most recent assistant message's last-iter prefill
	//     (prompt_n + cache_n). For agentic flows that's the LAST tool-loop
	//     iteration; for plain flows it's the single call. Matches
	//     contextUsed - last iter output, so the user can sanity-check the
	//     gauge against the breakdown.
	//   - cumulativeStats: sum across every assistant message × tool-loop
	//     iteration. For agentic messages prefer the agentic.llm rollup
	//     (covers every iter); for plain messages the top-level timings
	//     already describe the full turn. Avg speed uses this scope since
	//     it makes more sense as a run-wide throughput.
	//
	// Live processingState feeds both. For current we MAX the last persisted
	// value with the live reading because the in-flight iter hasn't been
	// written yet; for cumulative we ADD the live on top of the persisted
	// rollup (the rollup only sums COMPLETED iters per agentic.svelte.ts,
	// so there's no double-count with the live in-flight value).
	let currentStats = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];
		const live = processingState.processingState;
		const isLive = isLoading() || isChatStreaming();

		let read = 0;
		let output = 0;

		// Walk backward to the most recent assistant message with timings so
		// an active stream on a fresh message falls back to the prior turn;
		// the live override below still drives the value when nothing is
		// persisted yet.
		for (let i = messages.length - 1; i >= 0; i--) {
			const m = messages[i];
			if (m.role !== MessageRole.ASSISTANT) continue;
			const t = m.timings;
			if (!t) continue;
			// Top-level timings are the LAST iter's values regardless of
			// whether the agentic rollup is attached (buildFinalTimings sets
			// prompt_n from capturedTimings, see agentic.svelte.ts). The
			// agentic.llm fields sum across iters and would over-report here.
			read = (t.prompt_n ?? 0) + (t.cache_n ?? 0);
			output = t.predicted_n ?? 0;
			break;
		}

		if (isLive && live) {
			// promptTokens grows during the read phase, but SSE also emits
			// prompt_progress events separately from timings, so during the
			// pure reading phase promptTokens can be 0 even though reading is
			// actively progressing. fall back to promptProgress.processed in
			// that case (canonical server-reported read counter).
			const livePromptProgress = live.promptProgress?.processed ?? 0;
			const liveRead = Math.max(live.promptTokens ?? 0, livePromptProgress);
			if (liveRead > 0) read = Math.max(read, liveRead);
			const liveOut = live.outputTokensUsed ?? 0;
			if (liveOut > 0) output = Math.max(output, liveOut);
		}

		return { read, output };
	});

	let cumulativeStats = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];
		const live = processingState.processingState;
		const isLive = isLoading() || isChatStreaming();

		let read = 0;
		let output = 0;
		let outputMs = 0;

		for (const m of messages) {
			if (m.role !== MessageRole.ASSISTANT) continue;
			const t = m.timings;
			if (!t) continue;

			const llm = t.agentic?.llm;
			if (
				t.agentic &&
				(llm?.prompt_n != null || llm?.predicted_n != null || llm?.predicted_ms != null)
			) {
				read += llm?.prompt_n ?? 0;
				output += llm?.predicted_n ?? 0;
				outputMs += llm?.predicted_ms ?? 0;
			} else {
				read += t.prompt_n ?? 0;
				output += t.predicted_n ?? 0;
				outputMs += t.predicted_ms ?? 0;
			}
		}

		if (isLive && live) {
			const livePromptProgress = live.promptProgress?.processed ?? 0;
			const liveRead = Math.max(live.promptTokens ?? 0, livePromptProgress);
			if (liveRead > 0) read += liveRead;
			const liveOut = live.outputTokensUsed ?? 0;
			if (liveOut > 0) output += liveOut;
			const liveTps = live.tokensPerSecond ?? 0;
			if (liveTps > 0 && liveOut > 0) {
				outputMs += (liveOut / liveTps) * 1000;
			}
		}

		const averageTokensPerSecond =
			outputMs > 0 && output > 0 ? (output / outputMs) * 1000 : null;

		return { read, output, averageTokensPerSecond };
	});

	// Per-message output: do NOT show (replaced by currentStats.output /
	// cumulativeStats.output).
	// Per-message t/s: do NOT show (replaced by cumulativeStats.averageTokensPerSecond).
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

	let detailsOpen = $state(false);

	let hasDetails = $derived(
		enabledToolsTokenCount !== null && enabledToolsTokenCount > 0 ||
			currentStats.read > 0 ||
			cumulativeStats.read > 0 ||
			currentStats.output > 0 ||
			cumulativeStats.output > 0 ||
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
				class:text-amber-400={contextPercent !== null && contextPercent >= 80 && contextPercent < 95}
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
						class:bg-amber-500={contextPercent !== null && contextPercent >= 80 && contextPercent < 95}
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

					<Collapsible.Content class="flex flex-col gap-2 text-xs pt-4">
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
					{#if currentStats.read > 0}
						<div class="flex items-baseline justify-between">
							<span class="text-muted-foreground">Reading</span>
							<span class="font-mono text-muted-foreground">
								{currentStats.read.toLocaleString()} tok
							</span>
						</div>
						{#if cumulativeStats.read > currentStats.read}
							<div class="-mt-1.5 pl-2 text-[10px] leading-tight text-muted-foreground/70">
								{cumulativeStats.read.toLocaleString()} total across the conversation
							</div>
						{/if}
					{/if}
					{#if cumulativeStats.output > 0}
						<div class="flex items-baseline justify-between">
							<span class="text-muted-foreground">Output</span>
							<span class="font-mono text-muted-foreground">
								{cumulativeStats.output.toLocaleString()} tok
							</span>
						</div>
						{#if currentStats.output > 0 && currentStats.output < cumulativeStats.output}
							<div class="-mt-1.5 pl-2 text-[10px] leading-tight text-muted-foreground/70">
								{currentStats.output.toLocaleString()} in the last response
							</div>
						{/if}
					{/if}
					{#if cumulativeStats.averageTokensPerSecond !== null}
						<div class="flex items-baseline justify-between">
							<span class="text-muted-foreground">Avg speed</span>
							<span class="font-mono text-muted-foreground">
								{cumulativeStats.averageTokensPerSecond.toFixed(1)}
								{STATS_UNITS.TOKENS_PER_SECOND}
							</span>
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
