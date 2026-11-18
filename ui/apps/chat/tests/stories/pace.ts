import { userEvent } from 'storybook/test';

/** Toolbar values of the `interactionPace` global, see .storybook/preview.ts. */
export enum InteractionPace {
	HUMAN = 'human',
	INSTANT = 'instant'
}

export interface Pace {
	/** Delay between streamed reply chunks of a fake server. */
	streamChunkMs: number;
	/** Waits before the next action, as if reading the screen first. */
	pause: (ms?: number) => Promise<void>;
	/** A user-event instance that types key by key at the chosen pace. */
	user: ReturnType<typeof userEvent.setup>;
}

// the test runner drives a Playwright browser, which always plays stories at full speed
const isTestRun = typeof navigator !== 'undefined' && navigator.webdriver;
const HUMAN = { beforeActionMs: 900, streamChunkMs: 110, typingMs: 55 };
const INSTANT = { beforeActionMs: 0, streamChunkMs: 30, typingMs: 0 };

/**
 * Pace for a story's `beforeEach` or `play`: human in the Storybook UI (unless the
 * toolbar says instant), instant in the test suite.
 */
export function getPace(context: { globals?: Record<string, unknown> }): Pace {
	const human = !isTestRun && context.globals?.interactionPace !== InteractionPace.INSTANT;
	const timings = human ? HUMAN : INSTANT;

	return {
		pause: (ms = timings.beforeActionMs) =>
			ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve(),
		streamChunkMs: timings.streamChunkMs,
		user: userEvent.setup({ delay: timings.typingMs })
	};
}
