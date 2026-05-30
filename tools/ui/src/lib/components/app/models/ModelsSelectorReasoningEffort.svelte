<script lang="ts">
	import { Check, Info } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { ReasoningEffort } from '$lib/enums';
	import { conversationsStore } from '$lib/stores/conversations.svelte';

	const EFFORT_LEVELS = [
		{ value: ReasoningEffort.LOW, label: 'Low', default: true },
		{ value: ReasoningEffort.MEDIUM, label: 'Medium' },
		{ value: ReasoningEffort.HIGH, label: 'High' },
		{ value: ReasoningEffort.MAX, label: 'Max', hasInfo: true }
	];

	let currentEffort = $derived(conversationsStore.getReasoningEffort());
	let thinkingEnabled = $derived(conversationsStore.getThinkingEnabled());

	function handleEffortChange(effort: string) {
		conversationsStore.setReasoningEffort(effort);
	}

	function toggleThinking() {
		conversationsStore.setThinkingEnabled(!conversationsStore.getThinkingEnabled());
	}
</script>

<DropdownMenu.Root>
	<DropdownMenu.Trigger
		class="flex cursor-pointer items-center gap-1.5 rounded-sm px-1 py-0.5 text-xs text-muted-foreground transition hover:bg-muted-foreground/10 hover:text-foreground"
	>
		<span>{currentEffort.charAt(0).toUpperCase() + currentEffort.slice(1)}</span>

		<svg
			class="h-3 w-3 shrink-0"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			stroke-width="2"
		>
			<path d="m6 9 6 6 6-6" />
		</svg>
	</DropdownMenu.Trigger>

	<DropdownMenu.Content
		align="end"
		class="w-48 rounded-xl bg-popover p-2 text-popover-foreground shadow-md outline-none data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
	>
		<div class="mb-1.5 px-2 py-1 text-xs text-muted-foreground">
			Higher effort means more thorough responses, but takes longer and uses your limits faster.
		</div>

		{#each EFFORT_LEVELS as level (level.value)}
			<DropdownMenu.Item
				class="flex w-full cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-sm transition-colors outline-none hover:bg-accent focus:bg-accent"
				onclick={() => handleEffortChange(level.value)}
			>
				{#if currentEffort === level.value}
					<Check class="h-4 w-4 text-foreground" />
				{:else}
					<div class="h-4 w-4" />
				{/if}

				<span>{level.label}</span>

				{#if level.default}
					<span
						class="ml-auto rounded bg-muted-foreground/10 px-1.5 py-0.5 text-[11px] text-muted-foreground"
					>
						Default
					</span>
				{/if}

				{#if level.hasInfo}
					<Info class="ml-1 h-3.5 w-3.5 text-muted-foreground" />
				{/if}
			</DropdownMenu.Item>
		{/each}

		<div class="my-1.5 h-px bg-border" />

		<div class="px-2 py-1.5">
			<div class="text-sm font-medium">Thinking</div>

			<div class="mt-1 flex items-center justify-between">
				<span class="text-xs text-muted-foreground">Can think for more complex tasks</span>

				<button
					type="button"
					class="inline-flex h-4 w-9 shrink-0 cursor-pointer items-center rounded-full bg-muted-foreground/20 p-px transition-colors hover:bg-muted-foreground/30"
					class:bg-foreground={thinkingEnabled}
					onclick={toggleThinking}
				>
					<span
						class="block h-3 w-3 rounded-full bg-background transition-transform"
						class:translate-x-4={thinkingEnabled}
					/>
				</button>
			</div>
		</div>
	</DropdownMenu.Content>
</DropdownMenu.Root>
