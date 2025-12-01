<script lang="ts">
	import { Clock, Gauge, WholeWord } from '@lucide/svelte';
	import { BadgeChatStatistic } from '$lib/components/app';

	interface Props {
		predictedTokens: number;
		predictedMs: number;
	}

	let { predictedTokens, predictedMs }: Props = $props();

	let tokensPerSecond = $derived((predictedTokens / predictedMs) * 1000);
	let timeInSeconds = $derived((predictedMs / 1000).toFixed(2));
</script>

<BadgeChatStatistic icon={WholeWord} value="{predictedTokens} tokens" />

<BadgeChatStatistic icon={Clock} value="{timeInSeconds}s" />

<BadgeChatStatistic icon={Gauge} value="{tokensPerSecond.toFixed(2)} tokens/s" />
