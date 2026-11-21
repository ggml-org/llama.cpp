<!--
	The chat's reasoning effort on the closed model selector, next to the model: the level as a word when it is set to
	one, a crossed-out lightbulb when reasoning is off, and nothing at the default, which leaves it to the model.
-->
<script lang="ts">
	import { LightbulbOff } from '@lucide/svelte';
	import { REASONING_EFFORT_LABELS } from '#lib/constants/index.ts';
	import { ReasoningEffort } from '#lib/enums/index.ts';
	import { useReasoningMenu } from '#lib/hooks/use-reasoning-menu.svelte.ts';

	const reasoning = useReasoningMenu();
</script>

{#if reasoning.isOff}
	<span
		class="ml-1 flex shrink-0 items-center text-muted-foreground"
		data-slot="models-selector-reasoning-badge"
		title="Reasoning: Off"
	>
		<LightbulbOff aria-hidden="true" class="size-icon-xs max-md:size-icon-sm shrink-0" />

		<span class="sr-only">Reasoning off</span>
	</span>
{:else if reasoning.currentEffort !== ReasoningEffort.DEFAULT}
	<span
		class="ml-1 shrink-0 text-muted-foreground"
		data-slot="models-selector-reasoning-badge"
		title="Reasoning: {REASONING_EFFORT_LABELS[reasoning.currentEffort]}"
	>
		<span aria-hidden="true">{REASONING_EFFORT_LABELS[reasoning.currentEffort]}</span>

		<span class="sr-only">Reasoning {REASONING_EFFORT_LABELS[reasoning.currentEffort]}</span>
	</span>
{/if}
