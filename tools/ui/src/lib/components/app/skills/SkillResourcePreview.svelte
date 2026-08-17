<script lang="ts">
	import { Circle, RefreshCw } from '@lucide/svelte';
	import { MarkdownContent, SyntaxHighlightedCode } from '$lib/components/app';
	import { Alert, AlertDescription, AlertTitle } from '$lib/components/ui/alert';
	import { Button } from '$lib/components/ui/button';
	import { SkillsService } from '$lib/services/skills.service';
	import type { SkillBaseReadResult } from '$lib/types';
	import { ApiError, getLanguageFromFilename } from '$lib/utils';

	interface Props {
		baseResult: SkillBaseReadResult;
		cwd: string | undefined;
		selectedPath: string;
		onAvailabilityChange: (path: string, available: boolean) => void;
	}

	let { baseResult, cwd, selectedPath, onAvailabilityChange }: Props = $props();
	let readState = $state<'loading' | 'ready' | 'error'>('ready');
	let source = $state('');
	let errorMessage = $state('');
	let retryToken = $state(0);
	let generation = 0;

	const format = $derived(selectedPath === 'SKILL.md' ? 'markdown' : resourceFormat(selectedPath));
	const canRenderMarkdown = $derived(format === 'markdown');
	const canRenderHtml = $derived(format === 'html');
	let mode = $state<'rendered' | 'raw' | 'preview' | 'source'>('rendered');
	const iframeSource = $derived(`<!doctype html><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">${source}`);

	$effect(() => {
		void retryToken;
		const controller = new AbortController();
		const currentGeneration = ++generation;
		errorMessage = '';

		if (selectedPath === 'SKILL.md') {
			source = baseResult.source;
			readState = 'ready';
			return () => controller.abort();
		}

		readState = 'loading';
		source = '';
		SkillsService.read({ name: baseResult.skill.name, path: selectedPath }, cwd, controller.signal)
			.then((response) => {
				if (controller.signal.aborted || currentGeneration !== generation) return;

				if (
					response.kind !== 'resource' ||
					response.skill.id !== baseResult.skill.id ||
					response.resource.path !== selectedPath
				) {
					errorMessage = 'The server returned content for a different resource.';
					readState = 'error';
					return;
				}

				source = response.source;
				readState = 'ready';
			})
			.catch((failure) => {
				if (controller.signal.aborted || currentGeneration !== generation) return;

				if (failure instanceof ApiError && failure.status >= 400 && failure.status < 500) {
					onAvailabilityChange(selectedPath, false);
					errorMessage = 'Preview unavailable';
				} else {
					errorMessage = failure instanceof Error ? failure.message : 'The resource could not be read.';
				}

				readState = 'error';
			});

		return () => controller.abort();
	});

	function resourceFormat(path: string): 'html' | 'markdown' | 'source' {
		if (path.toLowerCase().endsWith('.html') || path.toLowerCase().endsWith('.htm')) return 'html';
		if (path.toLowerCase().endsWith('.md') || path.toLowerCase().endsWith('.markdown')) return 'markdown';
		return 'source';
	}

	$effect(() => {
		mode = canRenderHtml ? 'preview' : canRenderMarkdown ? 'rendered' : 'source';
	});
</script>

{#if canRenderMarkdown || canRenderHtml}
	<div class="flex shrink-0 items-center gap-1" data-testid="skill-resource-preview-modes">
		{#if canRenderMarkdown}
			<Button variant={mode === 'rendered' ? 'default' : 'ghost'} size="sm" onclick={() => (mode = 'rendered')}>Markdown</Button>
			<Button variant={mode === 'raw' ? 'default' : 'ghost'} size="sm" onclick={() => (mode = 'raw')}>Raw</Button>
		{:else}
			<Button variant={mode === 'preview' ? 'default' : 'ghost'} size="sm" onclick={() => (mode = 'preview')}>Preview</Button>
			<Button variant={mode === 'source' ? 'default' : 'ghost'} size="sm" onclick={() => (mode = 'source')}>Source</Button>
		{/if}
	</div>
{/if}

{#if readState === 'loading'}
	<p class="text-sm text-muted-foreground">Loading resource...</p>
{:else if readState === 'error'}
	<Alert variant="destructive">
		<Circle class="size-4" />
		<AlertTitle>Could not load the resource</AlertTitle>
		<AlertDescription>{errorMessage}</AlertDescription>
	</Alert>
	{#if errorMessage !== 'Preview unavailable'}
		<Button class="mt-4" onclick={() => (retryToken += 1)}><RefreshCw class="size-3" />Retry</Button>
	{/if}
{:else if canRenderMarkdown && mode === 'rendered'}
	<div data-testid={selectedPath === 'SKILL.md' ? 'skill-detail-markdown' : 'skill-resource-markdown'}><MarkdownContent content={selectedPath === 'SKILL.md' ? baseResult.body_markdown : source} /></div>
{:else if canRenderHtml && mode === 'preview'}
	<iframe title={selectedPath} sandbox="" srcdoc={iframeSource} class="h-full min-h-80 w-full rounded-md border bg-white" data-testid="skill-resource-html-preview"></iframe>
{:else}
	<div data-testid={selectedPath === 'SKILL.md' ? 'skill-detail-raw' : undefined}>
		<SyntaxHighlightedCode code={source} language={getLanguageFromFilename(selectedPath) || 'plaintext'} />
	</div>
{/if}
