<script lang="ts">
	import { Check, Info, Lightbulb, LightbulbOff } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { ReasoningEffort } from '$lib/enums';
	import { conversationsStore } from '$lib/stores/conversations.svelte';

	const EFFORT_LEVELS = [
		{ value: 'off', label: 'Off', isOff: true },
		{ value: ReasoningEffort.LOW, label: 'Low', default: true },
		{ value: ReasoningEffort.MEDIUM, label: 'Medium' },
		{ value: ReasoningEffort.HIGH, label: 'High' },
		{ value: ReasoningEffort.MAX, label: 'Max', hasInfo: true }
	];

	let thinkingEnabled = $derived(conversationsStore.getThinkingEnabled());
	let currentEffort = $derived(conversationsStore.getReasoningEffort());
	let isOff = $derived(!thinkingEnabled);

	// Check if current item is selected
	function isSelected(item: (typeof EFFORT_LEVELS)[number]): boolean {
		if (item.isOff) {
			return isOff;
		}
		return thinkingEnabled && currentEffort === item.value;
	}

	function handleSelection(item: (typeof EFFORT_LEVELS)[number]) {
		if (item.isOff) {
			conversationsStore.setThinkingEnabled(false);
		} else {
			conversationsStore.setThinkingEnabled(true);
			conversationsStore.setReasoningEffort(item.value as ReasoningEffort);
		}
	}
</script>

<DropdownMenu.Sub>
	<DropdownMenu.SubTrigger
		class="flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-sm transition-colors outline-none hover:bg-accent focus:bg-accent"
	>
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

	<DropdownMenu.SubContent
		class="w-56 rounded-xl bg-popover p-3 text-popover-foreground shadow-md outline-none data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
	>
		{#each EFFORT_LEVELS as level (level.value)}
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

				{#if level.default && !isOff}
					<span
						class="rounded bg-muted-foreground/10 px-1.5 py-0.5 text-[11px] text-muted-foreground"
					>
						Default
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
			</button>
		{/each}
	</DropdownMenu.SubContent>
</DropdownMenu.Sub>
