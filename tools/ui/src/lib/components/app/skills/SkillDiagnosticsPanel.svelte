<!-- tools/ui/src/lib/components/app/skills/SkillDiagnosticsPanel.svelte -->
<script lang="ts">
	import SkillDiagnosticRow from './SkillDiagnosticRow.svelte';
	import { ChevronDown, X } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app/actions';
	import { Badge } from '$lib/components/ui/badge';
	import type { SkillBudgetChip } from '$lib/utils/skill-budget-chip';
	import type { SkillDiagnostic } from '$lib/types';

	interface Props {
		diagnostics: readonly SkillDiagnostic[];
		budgetChip: SkillBudgetChip | null;
		dismissed: boolean;
		onDismiss: () => void;
	}

	let { budgetChip, diagnostics, dismissed, onDismiss }: Props = $props();

	const errors = $derived(diagnostics.filter((d) => d.severity === 'error'));
	const warnings = $derived(diagnostics.filter((d) => d.severity === 'warning'));
	const hasContent = $derived(errors.length > 0 || warnings.length > 0 || budgetChip !== null);

	let warningsExpanded = $state(false);

	// A fresh diagnostics list (new catalog) always starts collapsed again.
	$effect(() => {
		void diagnostics;
		warningsExpanded = false;
	});
</script>

{#if !dismissed && hasContent}
	<div class="relative flex flex-col gap-2 pr-6">
		{#if budgetChip}
			<div class="flex items-start gap-2 text-sm">
				<Badge
					variant="outline"
					class="w-fit shrink-0 border-amber-500/40 text-amber-700 dark:text-amber-400"
					title={budgetChip.detail}
				>
					{budgetChip.label}
				</Badge>
			</div>
		{/if}

		{#each errors as diagnostic, i (`${diagnostic.code}-error-${i}`)}
			<SkillDiagnosticRow {diagnostic} />
		{/each}

		{#if warnings.length > 1 && !warningsExpanded}
			<button
				type="button"
				data-testid="skill-diagnostics-warnings-toggle"
				class="inline-flex w-fit items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
				aria-expanded="false"
				onclick={() => (warningsExpanded = true)}
			>
				<Badge variant="outline" class="border-amber-500/40 text-amber-700 dark:text-amber-400">
					warning
				</Badge>
				{warnings.length} warnings
				<ChevronDown class="size-3.5" aria-hidden="true" />
			</button>
		{:else}
			{#each warnings as diagnostic, i (`${diagnostic.code}-warning-${i}`)}
				<SkillDiagnosticRow {diagnostic} />
			{/each}

			{#if warnings.length > 1}
				<button
					type="button"
					data-testid="skill-diagnostics-warnings-toggle"
					class="inline-flex w-fit items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
					aria-expanded="true"
					onclick={() => (warningsExpanded = false)}
				>
					Show less
					<ChevronDown class="size-3.5 rotate-180" aria-hidden="true" />
				</button>
			{/if}
		{/if}

		<ActionIcon
			icon={X}
			tooltip="Dismiss diagnostics"
			class="absolute right-2 top-2"
			onclick={onDismiss}
		/>
	</div>
{/if}
