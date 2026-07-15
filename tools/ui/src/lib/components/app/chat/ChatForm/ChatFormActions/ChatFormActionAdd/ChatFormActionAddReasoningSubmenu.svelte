<script lang="ts">
	import { Lightbulb, LightbulbOff, Check, Info } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { useReasoningMenu } from '$lib/hooks/use-reasoning-menu.svelte';

	let subOpen = $state(false);

	const reasoning = useReasoningMenu();
</script>

{#if reasoning.modelSupportsThinking}
	<DropdownMenu.Sub bind:open={subOpen}>
		<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
			{#if reasoning.thinkingEnabled}
				<Lightbulb class="h-4 w-4 shrink-0 text-amber-400" />
			{:else}
				<LightbulbOff class="h-4 w-4 shrink-0 text-muted-foreground" />
			{/if}

			<span
				class="text-sm inline-flex gap-2 {!reasoning.thinkingEnabled
					? 'text-muted-foreground'
					: ''}"
			>
				Reasoning

				<span class="capitalize text-muted-foreground">
					{reasoning.thinkingEnabled ? reasoning.currentEffort : 'off'}
				</span>
			</span>
		</DropdownMenu.SubTrigger>

		<DropdownMenu.SubContent
			class="w-60 bg-popover p-1.5 text-popover-foreground shadow-md outline-none"
		>
			{#each reasoning.levels as level (level.value)}
				{@const tokenLabel = reasoning.tokenLabel(level)}
				<button
					type="button"
					class="flex w-full cursor-pointer items-center gap-3 rounded-md px-2 py-1.75 text-left text-sm transition-colors hover:bg-accent"
					class:bg-accent={reasoning.isSelected(level)}
					onclick={() => {
						reasoning.select(level);
						subOpen = false;
					}}
				>
					{#if reasoning.isSelected(level)}
						<Check class="h-4 w-4 shrink-0 text-foreground" />
					{:else}
						<div class="h-4 w-4 shrink-0"></div>
					{/if}

					<span class="flex-1">{level.label}</span>

					{#if tokenLabel}
						<span class="text-[11px] text-muted-foreground opacity-60">
							{tokenLabel}
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
		</DropdownMenu.SubContent>
	</DropdownMenu.Sub>
{/if}
