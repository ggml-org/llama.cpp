<script lang="ts">
	import { Clock, Gauge, WholeWord, BookOpenText, Sparkles } from '@lucide/svelte';
	import { BadgeChatStatistic } from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		predictedTokens: number;
		predictedMs: number;
		promptTokens?: number;
		promptMs?: number;
	}

	let { predictedTokens, predictedMs, promptTokens, promptMs }: Props = $props();

	type StatsView = 'generation' | 'reading';
	let activeView: StatsView = $state('generation');

	let tokensPerSecond = $derived((predictedTokens / predictedMs) * 1000);
	let timeInSeconds = $derived((predictedMs / 1000).toFixed(2));

	let promptTokensPerSecond = $derived(
		promptTokens !== undefined && promptMs !== undefined
			? (promptTokens / promptMs) * 1000
			: undefined
	);
	let promptTimeInSeconds = $derived(
		promptMs !== undefined ? (promptMs / 1000).toFixed(2) : undefined
	);
	let hasPromptStats = $derived(
		promptTokens !== undefined &&
			promptMs !== undefined &&
			promptTokensPerSecond !== undefined &&
			promptTimeInSeconds !== undefined
	);
</script>

<div class="inline-flex items-center text-xs text-muted-foreground">
	<!-- Switcher triggers -->
	<div class="inline-flex items-center rounded-sm bg-muted-foreground/15 p-0.5">
		{#if hasPromptStats}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<button
						type="button"
						class="inline-flex h-5 w-5 items-center justify-center rounded-sm transition-colors {activeView ===
						'reading'
							? 'bg-background text-foreground shadow-sm'
							: 'hover:text-foreground'}"
						onclick={() => (activeView = 'reading')}
					>
						<BookOpenText class="h-3 w-3" />
						<span class="sr-only">Reading</span>
					</button>
				</Tooltip.Trigger>
				<Tooltip.Content>
					<p>Reading (prompt processing)</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}
		<Tooltip.Root>
			<Tooltip.Trigger>
				<button
					type="button"
					class="inline-flex h-5 w-5 items-center justify-center rounded-sm transition-colors {activeView ===
					'generation'
						? 'bg-background text-foreground shadow-sm'
						: 'hover:text-foreground'}"
					onclick={() => (activeView = 'generation')}
				>
					<Sparkles class="h-3 w-3" />
					<span class="sr-only">Generation</span>
				</button>
			</Tooltip.Trigger>
			<Tooltip.Content>
				<p>Generation (token output)</p>
			</Tooltip.Content>
		</Tooltip.Root>
	</div>

	<!-- Stats badges (ghost style) -->
	<div class="flex items-center gap-1 px-2">
		{#if activeView === 'generation'}
			<BadgeChatStatistic
				class="bg-transparent"
				icon={WholeWord}
				value="{predictedTokens} tokens"
				tooltip="Generated tokens"
			/>
			<BadgeChatStatistic
				class="bg-transparent"
				icon={Clock}
				value="{timeInSeconds}s"
				tooltip="Generation time"
			/>
			<BadgeChatStatistic
				class="bg-transparent"
				icon={Gauge}
				value="{tokensPerSecond.toFixed(2)} tokens/s"
				tooltip="Generation speed"
			/>
		{:else if hasPromptStats}
			<BadgeChatStatistic
				class="bg-transparent"
				icon={WholeWord}
				value="{promptTokens} tokens"
				tooltip="Prompt tokens"
			/>
			<BadgeChatStatistic
				class="bg-transparent"
				icon={Clock}
				value="{promptTimeInSeconds}s"
				tooltip="Prompt processing time"
			/>
			<BadgeChatStatistic
				class="bg-transparent"
				icon={Gauge}
				value="{promptTokensPerSecond!.toFixed(2)} tokens/s"
				tooltip="Prompt processing speed"
			/>
		{/if}
	</div>
</div>
