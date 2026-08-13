/**
 * SkillCatalog - read-only presentation of the server's GET /skills catalog.
 *
 * Renders only contract-safe fields (name, description, scope, provider,
 * instruction facts, estimate state, timestamp, bounded resource count) plus
 * safe diagnostics. Opaque server XML is never rendered, the selected CWD is
 * never displayed, and unavailable (missing `--skills` route) stays distinct
 * from a request error and from a server-empty catalog. A zero budget keeps
 * the catalog listed while noting that no prompt envelope is packed.
 */
<script lang="ts">
	import { AlertCircle, BookOpen, CircleSlash, Clock, FileText, Layers, RefreshCw } from '@lucide/svelte';
	import { Alert, AlertDescription, AlertTitle } from '$lib/components/ui/alert';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
	import * as Empty from '$lib/components/ui/empty';
	import { Skeleton } from '$lib/components/ui/skeleton';
	import { SETTINGS_KEYS, normalizeSkillBudget } from '$lib/constants';
	import { skillsStore } from '$lib/stores/skills.svelte';
	import { settingsStore } from '$lib/stores/settings.svelte';
	import { ApiError } from '$lib/utils/api-fetch';
	import type { SkillCatalogEntry, SkillDiagnostic } from '$lib/types';

	interface Props {
		cwd: string | undefined;
		onRetry: () => void;
	}

	let { cwd, onRetry }: Props = $props();

	const budget = $derived(normalizeSkillBudget(settingsStore.config[SETTINGS_KEYS.MAX_SKILL_BUDGET]));
	const slot = $derived(skillsStore.slotFor(cwd));
	const status = $derived(slot?.status ?? 'loading');
	const error = $derived(slot?.error);
	const isUnavailable = $derived(
		status === 'error' && error instanceof ApiError && error.status === 404
	);
	const catalog = $derived(slot?.catalog);
	const isEmpty = $derived(status === 'ready' && (catalog?.skills.length ?? 0) === 0);
	const isZeroBudget = $derived(status === 'ready' && !isEmpty && budget === 0);
	const errorMessage = $derived(
		error instanceof Error ? error.message : 'The catalog could not be loaded.'
	);

	function formatTimestamp(value: string | null): string {
		if (!value) return '—';

		const date = new Date(value);

		if (Number.isNaN(date.getTime())) return '—';

		return date.toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' });
	}

	function resourceLabel(entry: SkillCatalogEntry): string {
		return entry.resources.truncated ? `${entry.resources.count}+` : `${entry.resources.count}`;
	}
</script>

<div class="mx-auto flex min-h-full w-full max-w-4xl flex-col gap-4 p-4 md:p-8">
	<div class="flex items-center gap-2">
		<BookOpen class="h-5 w-5 md:h-6 md:w-6" />

		<h1 class="text-lg font-semibold md:text-2xl">Skills catalog</h1>
	</div>

	<p class="text-sm text-muted-foreground">
		Budget:
		{budget === 0 ? '0 (no Skills prompt is packed)' : `${budget.toLocaleString()} tokens`}
	</p>

	{#if status === 'loading'}
		<div aria-label="Loading skills catalog" class="flex flex-col gap-3">
			<Skeleton class="h-24 w-full" />
			<Skeleton class="h-24 w-full" />
			<Skeleton class="h-24 w-full" />
		</div>
		<p class="text-sm text-muted-foreground">Loading catalog…</p>
	{:else if status === 'error'}
		{#if isUnavailable}
			<Alert>
				<CircleSlash class="h-4 w-4" />
				<AlertTitle>Skills are not enabled</AlertTitle>
				<AlertDescription>
					The server does not expose a Skills catalog. Start it with the <code>--skills</code>
					flag to browse skills here.
				</AlertDescription>
			</Alert>
		{:else}
			<Alert variant="destructive">
				<AlertCircle class="h-4 w-4" />
				<AlertTitle>Could not load the Skills catalog</AlertTitle>
				<AlertDescription>{errorMessage}</AlertDescription>
			</Alert>

			<div class="flex justify-start">
				<Button onclick={onRetry}>
					<RefreshCw class="h-3 w-3" />
					Retry
				</Button>
			</div>
		{/if}
	{:else if isEmpty}
		<div class="flex flex-1 items-center justify-center py-16">
			<Empty.Root class="max-w-md">
				<Empty.Header>
					<Empty.Media variant="icon">
						<BookOpen />
					</Empty.Media>

					<Empty.Title>No skills found</Empty.Title>

					<Empty.Description>
						The server returned an empty catalog for this working directory.
					</Empty.Description>
				</Empty.Header>
			</Empty.Root>
		</div>
	{:else}
		{#if isZeroBudget}
			<Alert>
				<CircleSlash class="h-4 w-4" />
				<AlertTitle>Budget is 0</AlertTitle>
				<AlertDescription>
					The catalog is listed here but no Skills prompt envelope is packed into agentic runs
					and no Skills tools are registered.
				</AlertDescription>
			</Alert>
		{/if}

		{#if catalog && catalog.diagnostics.length > 0}
			<div class="flex flex-col gap-2">
				{#each catalog.diagnostics as diagnostic (diagnostic.code)}
					<div class="flex items-start gap-2 text-sm">
						<Badge
							variant={diagnostic.severity === 'error' ? 'destructive' : 'secondary'}
							class="shrink-0"
						>
							{diagnostic.severity}
						</Badge>

						<span class="min-w-0 text-muted-foreground">
							<code class="mr-1">{diagnostic.code}</code>
							{diagnostic.name ?? diagnostic.scope ?? diagnostic.provider ?? ''}
							{diagnostic.message}
						</span>
					</div>
				{/each}
			</div>
		{/if}

		<div class="flex flex-col gap-3">
			{#each catalog?.skills ?? [] as entry (entry.id)}
				<Card>
					<CardHeader class="flex-row items-start justify-between gap-2 space-y-0">
						<div class="flex flex-col gap-1">
							<CardTitle class="text-base">{entry.name}</CardTitle>

							<div class="flex flex-wrap items-center gap-1.5">
								<Badge variant="secondary">{entry.scope}</Badge>
								<Badge variant="outline">{entry.provider}</Badge>
								<Badge variant={entry.instruction.tokens_estimated ? 'tertiary' : 'outline'}>
									{entry.instruction.tokens_estimated ? 'estimated' : 'exact'}
								</Badge>
							</div>
						</div>
					</CardHeader>

					<CardContent class="flex flex-col gap-3">
						{#if entry.description}
							<p class="text-sm text-muted-foreground">{entry.description}</p>
						{/if}

						<div class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
							<span class="inline-flex items-center gap-1">
								<FileText class="h-3 w-3" />
								{entry.instruction.tokens.toLocaleString()} tokens
							</span>
							<span>{entry.instruction.lines.toLocaleString()} lines</span>
							<span>{entry.instruction.bytes.toLocaleString()} bytes</span>
							<span class="inline-flex items-center gap-1">
								<Clock class="h-3 w-3" />
								Modified: {formatTimestamp(entry.instruction.modified_at)}
							</span>
							<span class="inline-flex items-center gap-1">
								<Layers class="h-3 w-3" />
								Resources:
								{resourceLabel(entry)}
							</span>
						</div>
					</CardContent>
				</Card>
			{/each}
		</div>
	{/if}
</div>
