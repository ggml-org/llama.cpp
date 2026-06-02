<script lang="ts">
	import { Lightbulb, LightbulbOff, Check, Info } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ReasoningEffort, MessageRole } from '$lib/enums';
	import { REASONING_EFFORT_TOKENS } from '$lib/constants/reasoning-effort-tokens';
	import { REASONING_EFFORT_LEVELS } from '$lib/constants/reasoning-effort';
	import type { ReasoningEffortLevel } from '$lib/types';
	import {
		modelsStore,
		checkModelSupportsThinking,
		supportsThinking,
		propsCacheVersion,
		loadedModelIds
	} from '$lib/stores/models.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore, activeMessages } from '$lib/stores/conversations.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import type { DatabaseMessage } from '$lib/types/database';

	let thinkingEnabled = $derived(conversationsStore.getThinkingEnabled());
	let currentEffort = $derived(conversationsStore.getReasoningEffort());
	let isOff = $derived(!thinkingEnabled);
	let tooltipText = $derived(thinkingEnabled ? `${currentEffort} Reasoning` : 'Disabled Reasoning');
	let subOpen = $state(false);

	// Get conversation model from message history
	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);

	// Fallback: if model props aren't available, check if any assistant messages
	// for this model in the active conversation have reasoning content.
	let modelSupportsThinkingFromMessages = $derived.by(() => {
		const modelId = isRouterMode() ? modelsStore.selectedModelName || conversationModel : null;
		if (!modelId) return false;
		const messages = conversationsStore.activeMessages;
		return messages.some(
			(m: DatabaseMessage) =>
				m.role === MessageRole.ASSISTANT && m.model === modelId && !!m.reasoningContent
		);
	});

	// Check if model supports thinking. Primary: chat template from /props.
	// Fallback: message history (reasoning content in assistant messages).
	let modelSupportsThinking = $derived.by(() => {
		loadedModelIds();
		propsCacheVersion();

		if (isRouterMode()) {
			const modelId = modelsStore.selectedModelName || conversationModel;
			return checkModelSupportsThinking(modelId ?? '') || modelSupportsThinkingFromMessages;
		}

		// In non-router mode, use the built-in supportsThinking
		return supportsThinking() || modelSupportsThinkingFromMessages;
	});

	// Check if current item is selected
	function isSelected(item: ReasoningEffortLevel): boolean {
		if (item.isOff) {
			return isOff;
		}
		return thinkingEnabled && currentEffort === item.value;
	}

	function handleSelection(item: ReasoningEffortLevel) {
		if (item.isOff) {
			conversationsStore.setThinkingEnabled(false);
		} else {
			conversationsStore.setThinkingEnabled(true);
			conversationsStore.setReasoningEffort(item.value as ReasoningEffort);
		}
		subOpen = false;
	}
</script>

{#if modelSupportsThinking}
	<DropdownMenu.Root bind:open={subOpen}>
		<Tooltip.Root>
			<Tooltip.Trigger>
				<DropdownMenu.Trigger
					class={[
						'flex h-6 w-6 cursor-pointer items-center justify-center rounded-full p-0 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
						thinkingEnabled ? 'bg-amber-400/10 hover:bg-amber-400/20' : 'bg-muted'
					]}
					aria-label={`${tooltipText}. Click to configure.`}
				>
					{#if thinkingEnabled}
						<Lightbulb class="h-3 w-3 text-amber-400" />
					{:else}
						<LightbulbOff class="h-3 w-3 text-muted-foreground" />
					{/if}
				</DropdownMenu.Trigger>
			</Tooltip.Trigger>

			<Tooltip.Content>
				<p class="capitalize">{tooltipText}</p>
			</Tooltip.Content>
		</Tooltip.Root>

		<DropdownMenu.Content
			align="start"
			class="w-60 rounded-xl bg-popover p-3 text-popover-foreground shadow-md outline-none"
		>
			<div class="mb-2 px-2.5 text-sm font-medium">Reasoning effort</div>

			{#each REASONING_EFFORT_LEVELS as level (level.value)}
				<button
					type="button"
					class="flex w-full cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm transition-colors hover:bg-accent"
					class:bg-accent={isSelected(level)}
					onclick={() => handleSelection(level)}
				>
					{#if isSelected(level)}
						<Check class="h-4 w-4 shrink-0 text-foreground" />
					{:else}
						<div class="h-4 w-4 shrink-0"></div>
					{/if}

					<span class="flex-1">{level.label}</span>

					{#if !level.isOff}
						<span class="text-[11px] text-muted-foreground opacity-60">
							{REASONING_EFFORT_TOKENS[level.value] === -1
								? 'Unlimited'
								: `Max ${REASONING_EFFORT_TOKENS[level.value].toLocaleString()} tokens`}
						</span>
					{/if}

					{#if level.hasInfo}
						<Tooltip.Root>
							<Tooltip.Trigger>
								<Info class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
							</Tooltip.Trigger>
							<Tooltip.Content side="left">
								<p>Maximum reasoning effort with extended context usage</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{/if}
				</button>
			{/each}
		</DropdownMenu.Content>
	</DropdownMenu.Root>
{/if}
