<!-- tools/ui/src/lib/components/app/skills/SkillDiagnosticsPanel.svelte -->
<script lang="ts">
	import type { SkillBudgetChip } from './skill-presentation';
	import SkillProviderLabel from './SkillProviderLabel.svelte';
	import { ChevronDown, X } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app/actions';
	import { Badge } from '$lib/components/ui/badge';
	import * as Card from '$lib/components/ui/card';
	import type { SkillDiagnostic } from '$lib/types';
	import { slide } from 'svelte/transition';

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

{#snippet row(diagnostic: SkillDiagnostic, showSeverity = true)}
	<div class="flex items-start gap-2 text-sm">
		{#if showSeverity}
			<Badge
				variant={diagnostic.severity === 'error' ? 'destructive' : 'outline'}
				class="shrink-0 {diagnostic.severity === 'warning'
					? 'border-amber-500/40 text-amber-700 dark:text-amber-400'
					: ''}"
			>
				{diagnostic.severity}
			</Badge>
		{/if}

		<span class="min-w-0 text-muted-foreground">
			{#if diagnostic.name}
				<span class="mr-2">Skill: {diagnostic.name}</span>
			{/if}
			{#if diagnostic.scope}
				<span class="mr-2">Scope: {diagnostic.scope}</span>
			{/if}
			{#if diagnostic.provider}
				<span class="mr-2">Provider: <SkillProviderLabel provider={diagnostic.provider} /></span>
			{/if}
			{#if diagnostic.providers && diagnostic.providers.length > 0}
				<span class="mr-2"
					>Providers: {#each diagnostic.providers as provider, index (provider)}{#if index > 0}<span
								>,&#32;</span
							>{/if}<SkillProviderLabel {provider} />{/each}</span
				>
			{/if}
			{diagnostic.message}
		</span>
	</div>
{/snippet}

{#if !dismissed && hasContent}
	<Card.Root
		role="region"
		aria-label="Skill diagnostics"
		class="relative gap-2 rounded-lg p-3 pr-10 shadow-none"
	>
		{#if budgetChip || warnings.length > 1}
			<div data-testid="skill-diagnostics-summary-row" class="flex items-start gap-2 text-sm">
				{#if budgetChip}
					<Badge
						variant="outline"
						class="w-fit shrink-0 border-amber-500/40 text-amber-700 dark:text-amber-400"
						title={budgetChip.detail}
					>
						{budgetChip.label}
					</Badge>
				{/if}

				{#if warnings.length > 1}
					<button
						type="button"
						data-testid="skill-diagnostics-warnings-toggle"
						class="inline-flex w-fit items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
						aria-expanded={warningsExpanded}
						onclick={() => (warningsExpanded = !warningsExpanded)}
					>
						<Badge variant="outline" class="border-amber-500/40 text-amber-700 dark:text-amber-400">
							warning
						</Badge>
						{warnings.length} warnings
						<ChevronDown
							class="size-3.5 {warningsExpanded ? 'rotate-180' : ''}"
							aria-hidden="true"
						/>
					</button>
				{/if}
			</div>
		{/if}

		{#each errors as diagnostic, i (`${diagnostic.code}-error-${i}`)}
			{@render row(diagnostic)}
		{/each}

		{#if warnings.length <= 1}
			{#each warnings as diagnostic, i (`${diagnostic.code}-warning-${i}`)}
				{@render row(diagnostic)}
			{/each}
		{:else if warningsExpanded}
			<div class="flex flex-col gap-2" transition:slide={{ duration: 150 }}>
				{#each warnings as diagnostic, i (`${diagnostic.code}-warning-${i}`)}
					{@render row(diagnostic, false)}
				{/each}
			</div>
		{/if}

		<ActionIcon
			icon={X}
			tooltip="Dismiss diagnostics"
			class="absolute right-2 top-2"
			onclick={onDismiss}
		/>
	</Card.Root>
{/if}
