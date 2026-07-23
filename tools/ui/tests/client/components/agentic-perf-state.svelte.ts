// Shared reactive fixture state for the Tier 1 perf harness.
//
// The wrapper component reads this module directly rather than taking props,
// so the driver can mutate it without depending on how the test renderer
// forwards props.

import type { DatabaseMessage } from '$lib/types';

export const perfState = $state<{
	message: DatabaseMessage | null;
	toolMessages: DatabaseMessage[];
	isStreaming: boolean;
}>({
	message: null,
	toolMessages: [],
	isStreaming: true
});
