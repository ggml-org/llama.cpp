<!-- tools/ui/src/lib/components/app/skills/SkillDiagnosticRow.svelte -->
<script lang="ts">
	import SkillProviderLabel from './SkillProviderLabel.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import type { SkillDiagnostic } from '$lib/types';

	interface Props {
		diagnostic: SkillDiagnostic;
		showSeverity?: boolean;
	}

	let { diagnostic, showSeverity = true }: Props = $props();
</script>

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
