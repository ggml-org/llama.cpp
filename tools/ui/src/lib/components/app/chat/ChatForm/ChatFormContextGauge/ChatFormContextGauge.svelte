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

	let contextUsed = $derived(processingState.processingState?.contextUsed ?? 0);

	let currentStats = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];
		const liveProcessingState = processingState.processingState;
		const isStreaming = isLoading() || isChatStreaming();

		let read = 0;
		let output = 0;

		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i];
			if (message.role !== MessageRole.ASSISTANT) continue;
			const timings = message.timings;
			if (!timings) continue;
			read = (timings.prompt_n ?? 0) + (timings.cache_n ?? 0);
			output = timings.predicted_n ?? 0;
			break;
		}

		if (isStreaming && liveProcessingState) {
			const livePromptProgress = liveProcessingState.promptProgress?.processed ?? 0;
			const livePromptTokens = Math.max(liveProcessingState.promptTokens ?? 0, livePromptProgress);
			if (livePromptTokens > 0) read = Math.max(read, livePromptTokens);
			const liveOutputTokens = liveProcessingState.outputTokensUsed ?? 0;
			if (liveOutputTokens > 0) output = Math.max(output, liveOutputTokens);
		}

		return { read, output };
	});

	let cumulativeStats = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];
		const liveProcessingState = processingState.processingState;
		const isStreaming = isLoading() || isChatStreaming();

		let read = 0;
		let output = 0;
		let outputMs = 0;

		for (const message of messages) {
			if (message.role !== MessageRole.ASSISTANT) continue;
			const timings = message.timings;
			if (!timings) continue;

			const agenticLlm = timings.agentic?.llm;
			if (
				timings.agentic &&
				(agenticLlm?.prompt_n != null ||
					agenticLlm?.predicted_n != null ||
					agenticLlm?.predicted_ms != null)
			) {
				read += agenticLlm?.prompt_n ?? 0;
				output += agenticLlm?.predicted_n ?? 0;
				outputMs += agenticLlm?.predicted_ms ?? 0;
			} else {
				read += timings.prompt_n ?? 0;
				output += timings.predicted_n ?? 0;
				outputMs += timings.predicted_ms ?? 0;
			}
		}

		if (isStreaming && liveProcessingState) {
			const livePromptProgress = liveProcessingState.promptProgress?.processed ?? 0;
			const livePromptTokens = Math.max(liveProcessingState.promptTokens ?? 0, livePromptProgress);
			if (livePromptTokens > 0) read += livePromptTokens;
			const liveOutputTokens = liveProcessingState.outputTokensUsed ?? 0;
			if (liveOutputTokens > 0) output += liveOutputTokens;
			const liveTokensPerSecond = liveProcessingState.tokensPerSecond ?? 0;
			if (liveTokensPerSecond > 0 && liveOutputTokens > 0) {
				outputMs += (liveOutputTokens / liveTokensPerSecond) * 1000;
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
		untrack(() => {
			const enabledToolsForLLM = toolsStore.getEnabledToolsForLLM();
			void enabledToolsForLLM;
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
		(enabledToolsTokenCount !== null && enabledToolsTokenCount > 0) ||
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

					<Collapsible.Content class="flex flex-col gap-2 text-xs pt-4">
						{#if enabledToolsTokenCount !== null && enabledToolsTokenCount > 0}
							<ContextGaugeDetailRow
								label="Tool definitions"
								value={formatParameters(enabledToolsTokenCount)}
								subtitle="Sent on every turn, cached after the first"
							/>
						{/if}
						{#if currentStats.read > 0}
							<ContextGaugeDetailRow
								label="Reading"
								value={`${currentStats.read.toLocaleString()} tok`}
								subtitle={cumulativeStats.read > currentStats.read
									? `${cumulativeStats.read.toLocaleString()} total across the conversation`
									: undefined}
							/>
						{/if}
						{#if cumulativeStats.output > 0}
							<ContextGaugeDetailRow
								label="Output"
								value={`${cumulativeStats.output.toLocaleString()} tok`}
								subtitle={currentStats.output > 0 && currentStats.output < cumulativeStats.output
									? `${currentStats.output.toLocaleString()} in the last response`
									: undefined}
							/>
						{/if}
						{#if cumulativeStats.averageTokensPerSecond !== null}
							<ContextGaugeDetailRow
								label="Avg speed"
								value={`${cumulativeStats.averageTokensPerSecond.toFixed(1)}${STATS_UNITS.TOKENS_PER_SECOND}`}
							/>
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
