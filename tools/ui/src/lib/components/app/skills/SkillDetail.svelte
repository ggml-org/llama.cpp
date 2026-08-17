<script lang="ts">
	import { Circle, RefreshCw, X } from '@lucide/svelte';
	import SkillResourcePicker from '$lib/components/app/skills/SkillResourcePicker.svelte';
	import SkillResourcePreview from '$lib/components/app/skills/SkillResourcePreview.svelte';
	import { ActionIcon } from '$lib/components/app/actions';
	import SkillProviderLabel from '$lib/components/app/skills/SkillProviderLabel.svelte';
	import { Alert, AlertDescription, AlertTitle } from '$lib/components/ui/alert';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Skeleton } from '$lib/components/ui/skeleton';
	import { SkillsService } from '$lib/services/skills.service';
	import type { SkillBaseReadResult, SkillCatalogEntry } from '$lib/types';
	import { fly } from 'svelte/transition';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		entry: SkillCatalogEntry;
		cwd: string | undefined;
		onClose: () => void;
		mobile: boolean;
	}

	let { cwd, entry, mobile, onClose }: Props = $props();

	// Ignore stale or aborted responses.
	let readState = $state<'loading' | 'ready' | 'error'>('loading');
	let result = $state<SkillBaseReadResult | null>(null);
	let errorMessage = $state('');
	let retryToken = $state(0);
	let readGeneration = 0;

	let selectedPath = $state('SKILL.md');
	let unavailablePaths = new SvelteSet<string>();

	// Reload when the entry or retry token changes.
	$effect(() => {
		void entry.id;
		void retryToken;
		const controller = new AbortController();
		const generation = ++readGeneration;

		result = null;
		selectedPath = 'SKILL.md';
		unavailablePaths.clear();
		errorMessage = '';

		SkillsService.read({ name: entry.name }, cwd, controller.signal)
			.then((readResult) => {
				if (generation !== readGeneration || controller.signal.aborted) return;

				// Preview accepts only base skill responses.
				if (readResult.kind !== 'skill') {
					errorMessage = 'The server returned a resource instead of the skill base content.';
					readState = 'error';

					return;
				}

				result = readResult;
				readState = 'ready';
			})
			.catch((failure) => {
				if (generation !== readGeneration || controller.signal.aborted) return;

				errorMessage = failure instanceof Error ? failure.message : 'The skill could not be read.';
				readState = 'error';
			});

		return () => controller.abort();
	});

	function retry() {
		retryToken += 1;
	}

	function setResourceAvailability(path: string, available: boolean) {
		if (available) unavailablePaths.delete(path);
		else unavailablePaths.add(path);
	}
</script>

<div
	data-testid="skill-detail"
	class="flex p-6 h-full min-h-0 flex-col"
	in:fly|global={{ duration: 200, opacity: 0, x: mobile ? 0 : 48 }}
>
	<div data-testid="skill-detail-header" class="flex shrink-0 flex-col gap-4">
		<div class="flex items-start justify-between gap-2">
			<div class="min-w-0">
				<h2 class="text-base font-semibold">{entry.name}</h2>
				<p class="text-xs text-muted-foreground">
					{entry.scope} / <SkillProviderLabel provider={entry.provider} />
				</p>
				{#if entry.disable_model_invocation}
					<Badge
						variant="secondary"
						class="mt-1"
						title="Not available to the model; activate with /skills <name> or from this catalog."
					>
						Manual only
					</Badge>
				{/if}
			</div>

			<ActionIcon icon={X} tooltip={mobile ? 'Back' : 'Close'} onclick={onClose} />
		</div>

		{#if result}
			<SkillResourcePicker
				paths={result.resources.paths}
				resourceCount={result.resources.paths.length}
				resourcesTruncated={result.resources.truncated}
				{selectedPath}
				{unavailablePaths}
				onSelect={(path) => (selectedPath = path)}
			/>
		{/if}
	</div>

	<div data-testid="skill-detail-separator" class="border-t mt-4" aria-hidden="true"></div>

	<div
		data-testid="skill-detail-body"
		class="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pt-4"
	>
		{#if readState === 'loading'}
			<div aria-label="Loading skill content" class="flex flex-col gap-3">
				<Skeleton class="h-8 w-40" />
				<Skeleton class="h-24 w-full" />
				<Skeleton class="h-24 w-full" />
			</div>
			<p class="text-sm text-muted-foreground">Loading skill content...</p>
		{:else if readState === 'error'}
			<Alert variant="destructive">
				<Circle class="h-4 w-4" />
				<AlertTitle>Could not load the skill</AlertTitle>
				<AlertDescription>{errorMessage}</AlertDescription>
			</Alert>

			<div class="flex justify-start">
				<Button onclick={retry}>
					<RefreshCw class="h-3 w-3" />
					Retry
				</Button>
			</div>
		{:else if result}
			<SkillResourcePreview
				baseResult={result}
				{cwd}
				{selectedPath}
				onAvailabilityChange={setResourceAvailability}
			/>
		{/if}
	</div>
</div>
