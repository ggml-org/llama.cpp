<script lang="ts">
	import { fade, fly } from 'svelte/transition';
	import { ChevronDown, Circle, RefreshCw, X } from '@lucide/svelte';
	import { MarkdownContent, SyntaxHighlightedCode } from '$lib/components/app';
	import { ActionIcon } from '$lib/components/app/actions';
	import { Alert, AlertDescription, AlertTitle } from '$lib/components/ui/alert';
	import { Button } from '$lib/components/ui/button';
	import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '$lib/components/ui/collapsible';
	import { Skeleton } from '$lib/components/ui/skeleton';
	import { FileTypeText } from '$lib/enums/files.enums';
	import { groupSkillResourcePaths } from '$lib/components/app/skills/skill-resource-presentation';
	import { SkillsService } from '$lib/services/skills.service';
	import type { SkillBaseReadResult, SkillCatalogEntry } from '$lib/types';

	interface Props {
		entry: SkillCatalogEntry;
		cwd: string | undefined;
		onClose: () => void;
		mobile: boolean;
	}

	let { entry, cwd, onClose, mobile }: Props = $props();

	/**
	 * The read lifecycle is owned by one cleanup-returning effect keyed on the
	 * entry and the selected CWD (plus a retry token). Every run takes a fresh
	 * AbortController and bumps a generation counter; the previous controller
	 * is aborted by the effect cleanup before the next run starts. A settling
	 * response is accepted only when it is still the current generation AND
	 * its own controller was not aborted, so a superseded, CWD-changed,
	 * retried, or unmounted read can never render. The retry replaces only the
	 * in-flight request.
	 */
	let readState = $state<'loading' | 'ready' | 'error'>('loading');
	let result = $state<SkillBaseReadResult | null>(null);
	let errorMessage = $state('');
	let retryToken = $state(0);
	let readGeneration = 0;

	// Each newly selected skill starts in the rendered markdown mode.
	let mode = $state<'markdown' | 'raw'>('markdown');

	let resourcesOpen = $state(true);

	$effect(() => {
		void entry.id;
		mode = 'markdown';
	});

	$effect(() => {
		void retryToken;
		const controller = new AbortController();
		const generation = ++readGeneration;

		readState = 'loading';
		result = null;
		errorMessage = '';

		SkillsService.read({ name: entry.name }, cwd, controller.signal)
			.then((readResult) => {
				if (generation !== readGeneration || controller.signal.aborted) return;

				// Preview accepts base results only; a resource result is an
				// error, never a content fallback.
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

				errorMessage =
					failure instanceof Error ? failure.message : 'The skill could not be read.';
				readState = 'error';
			});

		return () => controller.abort();
	});

	function retry() {
		retryToken += 1;
	}

	// Discovery is derived only from the successful base read; nothing else
	// renders or requests resources for the preview.
	const resourceGroups = $derived(
		result ? groupSkillResourcePaths(result.resources.paths) : []
	);
</script>

<div
	data-testid="skill-detail"
	class="flex p-6 h-full min-h-0 flex-col"
	in:fly|global={{ x: mobile ? 0 : 48, duration: 200, opacity: 0 }}
>
	<div data-testid="skill-detail-header" class="flex shrink-0 flex-col gap-4">
		<div class="flex items-start justify-between gap-2">
			<div class="min-w-0">
				<h2 class="text-base font-semibold">{entry.name}</h2>
				<p class="text-xs text-muted-foreground">
					{entry.scope} / {entry.provider}
				</p>
			</div>

			<div class="flex shrink-0 items-center gap-1">
				<Button
					variant={mode === 'markdown' ? 'default' : 'ghost'}
					size="sm"
					aria-pressed={mode === 'markdown'}
					onclick={() => (mode = 'markdown')}
				>
					Markdown
				</Button>
				<Button
					variant={mode === 'raw' ? 'default' : 'ghost'}
					size="sm"
					aria-pressed={mode === 'raw'}
					onclick={() => (mode = 'raw')}
				>
					Raw
				</Button>
				<ActionIcon icon={X} tooltip={mobile ? 'Back' : 'Close'} onclick={onClose} />
			</div>
		</div>

		{#if result}
			{#if resourceGroups.length > 0}
				<Collapsible
					bind:open={resourcesOpen}
					data-testid="skill-detail-resources"
					class="rounded-md border"
				>
					<CollapsibleTrigger
						class="flex cursor-pointer w-full items-center justify-between px-3 py-2 text-sm font-medium"
					>
						Resources ({result.resources.paths.length})
						<ChevronDown
							class="size-4 transition-transform duration-200 {resourcesOpen ? 'rotate-180' : ''}"
							aria-hidden="true"
						/>
					</CollapsibleTrigger>
					<CollapsibleContent
						class="overflow-hidden px-3 pb-3 data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down"
					>
						<div class="flex flex-col gap-3">
							{#each resourceGroups as resourceGroup (resourceGroup.group)}
								{@const ResourceIcon = resourceGroup.icon}
								<section
									data-testid="skill-resource-group"
									class="flex flex-col gap-1"
								>
									<div
										class="flex items-center gap-2 text-xs font-medium text-muted-foreground"
									>
										<ResourceIcon class="size-3.5" aria-hidden="true" />
										{resourceGroup.label}
									</div>
									{#each resourceGroup.paths as path (path)}
										<div class="pl-5 font-mono text-xs break-all">{path}</div>
									{/each}
								</section>
							{/each}
						</div>
					</CollapsibleContent>
				</Collapsible>
			{/if}
		{/if}
	</div>

	<div data-testid="skill-detail-separator" class="border-t mt-4" aria-hidden="true"></div>

	<div data-testid="skill-detail-body" class="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pt-4">
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
			{#if mode === 'markdown'}
				<div data-testid="skill-detail-markdown" class="min-w-0" in:fade={{ duration: 150 }}>
					<MarkdownContent content={result.body_markdown} />
				</div>
			{:else}
				<div
					data-testid="skill-detail-raw"
					class="min-w-0 overflow-y-hidden"
					in:fade={{ duration: 150 }}
				>
					<SyntaxHighlightedCode code={result.source} language={FileTypeText.MARKDOWN} />
				</div>
			{/if}
		{/if}
	</div>
</div>
