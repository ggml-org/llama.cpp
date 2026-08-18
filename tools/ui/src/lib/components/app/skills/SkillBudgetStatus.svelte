<script lang="ts">
	import { BookOpen, CircleSlash, Files, X } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app/actions';
	import { Alert, AlertDescription, AlertTitle } from '$lib/components/ui/alert';
	import type { SkillPackedCatalog } from '$lib/types';

	interface Props {
		packed: SkillPackedCatalog;
		budget: number;
		onDismiss?: () => void;
	}

	let { budget, onDismiss, packed }: Props = $props();

	const disabled = $derived(budget === 0 || packed.fullTokens === null);
	const complete = $derived(!disabled && packed.included === packed.total);
	// Keep the label nullable-safe for template branches.
	const fullTokensLabel = $derived(packed.fullTokens?.toLocaleString() ?? '');
	const measureLabel = $derived(packed.estimated ? 'estimated' : 'exact');
</script>

<div class="relative">
	{#if disabled}
		<Alert variant="warning">
			<CircleSlash class="h-4 w-4" />
			<AlertTitle>Skills tools are disabled</AlertTitle>
			<AlertDescription>
				Skills tools are disabled because the catalog budget is 0 tokens. The catalog stays
				available for browsing, but no Skills prompt envelope is packed into agentic runs.
			</AlertDescription>
		</Alert>
	{:else if complete}
		<Alert variant="success">
			<BookOpen class="h-4 w-4" />
			<AlertTitle>The full catalog fits the budget</AlertTitle>
			<AlertDescription>
				The full Skills catalog uses {fullTokensLabel} of {budget.toLocaleString()} budget tokens ({measureLabel}).
				list_skill() is not registered; read_skill() covers the full catalog.
			</AlertDescription>
		</Alert>
	{:else}
		<Alert variant="warning">
			<Files class="h-4 w-4" />
			<AlertTitle>The full catalog exceeds the budget</AlertTitle>
			<AlertDescription>
				The full Skills catalog requires {fullTokensLabel} tokens ({measureLabel}). {packed.included}
				of {packed.total} skills are included; list_skill() is available.
			</AlertDescription>
		</Alert>
	{/if}

	{#if onDismiss}
		<ActionIcon icon={X} tooltip="Dismiss" class="absolute right-2 top-2" onclick={onDismiss} />
	{/if}
</div>
