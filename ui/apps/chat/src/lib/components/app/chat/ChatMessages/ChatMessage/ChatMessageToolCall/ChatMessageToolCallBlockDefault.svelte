<script lang="ts">
	// Fall-through renderer for tool calls without a dedicated block.
	// Renders section.toolArgs / section.toolResult directly using the
	// shared chrome shell.

	import ToolCallBlock from './ToolCallBlock.svelte';
	import ToolCallTitle from './ToolCallTitle.svelte';
	import { simpleToolArgFields } from '@ggml-org/llama-harness';
	import { Loader2 } from '@lucide/svelte';
	import { MarkdownContent, SyntaxHighlightedCode } from '#lib/components/app/index.ts';
	import { MAX_HEIGHT_CODE_BLOCK } from '#lib/constants/index.ts';
	import { AttachmentType, FileTypeText, MimeTypeAudio, ToolResultKind } from '#lib/enums/index.ts';
	import type { AgenticSection, DatabaseMessageExtra, ToolResultLine } from '#lib/types/index.ts';
	import { createBase64DataUrl } from '#lib/utils/data-url.ts';
	import {
		classifyToolResult,
		formatJsonPretty,
		parseToolResultWithMedia
	} from '#lib/utils/index.ts';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		attachments?: DatabaseMessageExtra[];
		onToggle?: () => void;
	}

	let { attachments, isStreaming, onToggle, open, section }: Props = $props();

	const argFields = $derived(simpleToolArgFields(section.toolArgs));
	const outputKind = $derived(classifyToolResult(section.toolResult));
	const parsedLines: ToolResultLine[] = $derived(
		section.toolResult ? parseToolResultWithMedia(section.toolResult, attachments) : []
	);
</script>

{#snippet inputFields(fields: { label: string; value: string }[])}
	<dl class="grid grid-cols-[minmax(0,max-content)_minmax(0,1fr)] gap-x-6 gap-y-1 text-caption">
		{#each fields as field, i (i)}
			<dt class="text-muted-foreground">{field.label}</dt>

			<dd class="min-w-0 font-mono break-words text-foreground/90">{field.value}</dd>
		{/each}
	</dl>
{/snippet}

<ToolCallBlock {isStreaming} meta={null} {onToggle} {open} {section}>
	{#snippet titleSnippet(phrase)}
		<ToolCallTitle
			label={phrase.label}
			target={phrase.target && phrase.quoted ? `'${phrase.target}'` : phrase.target}
		/>
	{/snippet}

	{#snippet children(_meta, ctx)}
		{#if ctx.isStreamingCall}
			<div class="mb-2 flex items-center gap-2 text-caption text-muted-foreground/70">
				<span>Input</span>

				{#if ctx.isStreaming}
					<Loader2 class="size-icon-xs animate-spin" />
				{/if}
			</div>

			{#if section.toolArgs}
				<SyntaxHighlightedCode
					code={formatJsonPretty(section.toolArgs)}
					language={FileTypeText.JSON}
					maxHeight={MAX_HEIGHT_CODE_BLOCK}
					streaming={ctx.isCodeStreaming}
				/>
			{:else if ctx.isStreaming}
				<div class="rounded bg-muted/20 p-2 text-caption text-muted-foreground/70 italic">
					Receiving arguments...
				</div>
			{:else}
				<div
					class="rounded bg-yellow-500/10 p-2 text-caption text-yellow-600 italic dark:text-yellow-400"
				>
					Response was truncated
				</div>
			{/if}
		{:else}
			{@const showInput = Boolean(section.toolArgs)}
			{#if showInput}
				<div class="mb-1.5 flex items-center gap-2 text-caption text-muted-foreground/70">
					<span>Input</span>
				</div>

				{#if argFields}
					{@render inputFields(argFields)}
				{:else}
					<SyntaxHighlightedCode
						code={formatJsonPretty(section.toolArgs ?? '')}
						language={FileTypeText.JSON}
						maxHeight={MAX_HEIGHT_CODE_BLOCK}
						streaming={ctx.isCodeStreaming}
					/>
				{/if}
			{/if}

			<div
				class={showInput
					? 'mt-4 mb-1.5 flex items-center gap-2 text-caption text-muted-foreground/70'
					: 'mb-1.5 flex items-center gap-2 text-caption text-muted-foreground/70'}
			>
				<span>Output</span>

				{#if ctx.isPending}
					<Loader2 class="size-icon-xs animate-spin" />
				{/if}
			</div>

			{#if ctx.isPending}
				<div class="rounded bg-muted/20 p-2 text-caption text-muted-foreground/70 italic">
					Waiting for result...
				</div>
			{:else if section.toolResult}
				{#if outputKind === ToolResultKind.JSON}
					<SyntaxHighlightedCode
						code={formatJsonPretty(section.toolResult)}
						language={FileTypeText.JSON}
						maxHeight={MAX_HEIGHT_CODE_BLOCK}
					/>
				{:else if outputKind === ToolResultKind.MARKDOWN}
					<MarkdownContent {attachments} content={section.toolResult} />
				{:else}
					<div class="overflow-auto">
						{#each parsedLines as line, i (i)}
							<div class="font-mono text-caption leading-relaxed whitespace-pre-wrap">
								{line.text}
							</div>

							{#if line.media}
								{#if line.media.type === AttachmentType.AUDIO}
									{@const audioMimeType = line.media.mimeType ?? MimeTypeAudio.MP3_MPEG}
									<div class="mt-2 mb-2">
										<audio class="w-full rounded-lg" controls>
											<source
												src={createBase64DataUrl(audioMimeType, line.media.base64Data)}
												type={audioMimeType}
											/>
											Your browser does not support the audio element.
										</audio>
									</div>
								{:else}
									<img
										alt={line.media.name}
										class="mt-2 mb-2 h-auto max-w-full rounded-lg"
										loading="lazy"
										src={line.media.base64Url}
									/>
								{/if}
							{/if}
						{/each}
					</div>
				{/if}
			{:else}
				<div class="rounded bg-muted/20 p-2 text-caption text-muted-foreground/70 italic">
					No output
				</div>
			{/if}
		{/if}
	{/snippet}
</ToolCallBlock>
