<script lang="ts">
	import { Lightbulb, LightbulbOff } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import {
		checkModelSupportsThinking,
		propsCacheVersion,
		loadedModelIds
	} from '$lib/stores/models.svelte';
	import type { DatabaseMessage } from '$lib/types/database';
	import { MessageRole } from '$lib/enums';

	interface Props {
		modelId?: string | null;
	}

	let { modelId = null }: Props = $props();

	let enabled = $derived(conversationsStore.getThinkingEnabled());

	// Fallback: if model props aren't available, check if any assistant messages
	// for this model in the active conversation have reasoning content.
	let modelSupportsThinkingFromMessages = $derived.by(() => {
		if (!modelId) return false;
		const messages = conversationsStore.activeMessages;
		return messages.some(
			(m: DatabaseMessage) =>
				m.role === MessageRole.ASSISTANT && m.model === modelId && !!m.reasoningContent
		);
	});

	// Primary: check via chat template from /props. Falls back to message history.
	// Reads loadedModelIds + propsCacheVersion for reactivity when model loads and props cache updates.
	let modelSupportsThinking = $derived.by(() => {
		loadedModelIds();
		propsCacheVersion();
		const fromProps = checkModelSupportsThinking(modelId ?? '');
		return fromProps || modelSupportsThinkingFromMessages;
	});
</script>

{#if modelSupportsThinking}
	<Tooltip.Root>
		<Tooltip.Trigger>
			<span class="h-3 w-3 shrink-0 cursor-default rounded">
				{#if enabled}
					<Lightbulb class="h-3 w-3 fill-amber-400 text-amber-400" />
				{:else}
					<LightbulbOff class="h-3 w-3 fill-muted-foreground text-muted-foreground" />
				{/if}
			</span>
		</Tooltip.Trigger>
		<Tooltip.Content>
			<p>Thinking {enabled ? 'on' : 'off'}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{/if}
