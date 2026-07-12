<script lang="ts">
	import { Check, Info, Lightbulb, LightbulbOff } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ReasoningEffort, MessageRole } from '$lib/enums';
	import { REASONING_EFFORT_TOKENS } from '$lib/constants/reasoning-effort-tokens';
	import { REASONING_EFFORT_LEVELS } from '$lib/constants/reasoning-effort';
	import type { ReasoningEffortLevel } from '$lib/types';
	import { DIALOG_SUBMENU_CONTENT } from '$lib/constants/css-classes';
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
	let subOpen = $state(false);

	// Get conversation model from message history
	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);

	let modelSupportsThinkingFromMessages = $derived.by(() => {
		const modelId = isRouterMode() ? modelsStore.selectedModelName || conversationModel : null;
		if (!modelId) return false;

		const messages = conversationsStore.activeMessages;

		return messages.some(
			(m: DatabaseMessage) =>
				m.role === MessageRole.ASSISTANT && m.model === modelId && !!m.reasoningContent
		);
	});

	let modelSupportsThinking = $derived.by(() => {
		loadedModelIds();
		propsCacheVersion();

		if (isRouterMode()) {
			const modelId = modelsStore.selectedModelName || conversationModel;
			return checkModelSupportsThinking(modelId ?? '') || modelSupportsThinkingFromMessages;
		}

		return supportsThinking() || modelSupportsThinkingFromMessages;
	});

	function isSelected(item: ReasoningEffortLevel): boolean {
		if (item.isOff) return isOff;

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
	<DropdownMenu.Sub bind:open={subOpen}>
		<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
			{#if thinkingEnabled}
				<Lightbulb class="h-4 w-4 shrink-0 text-amber-400" />
			{:else}
				<LightbulbOff class="h-4 w-4 shrink-0 text-muted-foreground" />
			{/if}

			<span class="flex-1">Thinking</span>

			{#if thinkingEnabled}
				<span class="text-xs text-muted-foreground">{currentEffort}</span>
			{:else}
				<span class="text-xs text-muted-foreground">off</span>
			{/if}
		</DropdownMenu.SubTrigger>

		<DropdownMenu.SubContent class={DIALOG_SUBMENU_CONTENT}>
			{#each REASONING_EFFORT_LEVELS as level (level.value)}
				<button
					type="button"
					class="flex w-full cursor-pointer items-center gap-2"
					class:bg-accent={isSelected(level)}
					onclick={() => handleSelection(level)}
				>
					<span class="flex-1 text-left">{level.label}</span>

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
								<p>Maximum thinking effort with extended context usage</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{/if}

					{#if isSelected(level)}
						<Check class="h-4 w-4 shrink-0 text-foreground" />
					{/if}
				</button>
			{/each}
		</DropdownMenu.SubContent>
	</DropdownMenu.Sub>
{/if}
