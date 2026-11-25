// Guards the task time in a reply's statistics: the whole reply, from the
// prompt to the answer, ticking while it runs and as recorded once done.

import ChatMessageStatistics from './components/ChatMessageStatisticsTestWrapper.svelte';
import { ChatMessageStatisticsMode } from '#lib/enums/index.ts';
import type { ChatMessageAgenticTimings } from '#lib/types/index.ts';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const GENERATION = {
	mode: ChatMessageStatisticsMode.GENERATION,
	predictedMs: 2000,
	predictedTokens: 100
};
const LLM = { predicted_ms: 2000, predicted_n: 100, prompt_ms: 500, prompt_n: 50 };

function agentic(totalMs?: number): ChatMessageAgenticTimings {
	return { llm: LLM, toolCallsCount: 1, toolsMs: 300, totalMs, turns: 2 };
}

describe('task time statistics', () => {
	it('ticks from the task start while the reply runs', async () => {
		const { container } = render(ChatMessageStatistics, {
			...GENERATION,
			isLive: true,
			taskStartedAt: Date.now() - 5000
		});

		await tick();

		expect(container.textContent).toMatch(/\b5s\b/);
	});

	it('shows the recorded total once the reply is done', async () => {
		const { container } = render(ChatMessageStatistics, {
			...GENERATION,
			agenticTimings: agentic(12_345)
		});

		await tick();

		expect(container.textContent).toMatch(/\b12s\b/);
	});

	it('shows no task time when none was recorded', async () => {
		const { container } = render(ChatMessageStatistics, {
			...GENERATION,
			agenticTimings: agentic()
		});

		await tick();

		// tokens, generation time and speed only
		expect(container.textContent).not.toContain('2.8s');
		expect(container.querySelectorAll('svg.lucide-hourglass')).toHaveLength(0);
	});
});
