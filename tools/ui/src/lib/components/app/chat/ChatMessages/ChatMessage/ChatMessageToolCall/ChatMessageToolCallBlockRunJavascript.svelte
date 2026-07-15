<script lang="ts">
	import { XCircle, Terminal } from '@lucide/svelte';
	import { SyntaxHighlightedCode } from '$lib/components/app';
	import { FileTypeText } from '$lib/enums';
	import { MAX_HEIGHT_CODE_BLOCK } from '$lib/constants';
	import { getBuiltinToolUi, type AgenticSection } from '$lib/utils';
	import { parseRunJavascriptMeta } from './parsers/run-javascript';
	import ToolCallBlock from './ToolCallBlock.svelte';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { section, open, isStreaming, onToggle }: Props = $props();

	const runJsMeta = $derived(parseRunJavascriptMeta(section));
	const title = $derived(getBuiltinToolUi(section.toolName)?.label ?? section.toolName ?? '');
</script>

<ToolCallBlock {section} {open} {isStreaming} meta={runJsMeta} {title} {onToggle}>
	{#snippet children(meta, ctx)}
		{#if ctx.isPending}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">Running...</div>
		{:else if meta?.errorMessage}
			<div
				class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
			>
				<XCircle class="mt-0.5 h-3 w-3 shrink-0" />
				<span>{meta.errorMessage}</span>
			</div>
			<div class="mt-3">
				<SyntaxHighlightedCode
					code={meta.code}
					language={FileTypeText.JAVASCRIPT}
					maxHeight={MAX_HEIGHT_CODE_BLOCK}
					streaming={ctx.isCodeStreaming}
				/>
			</div>
		{:else if meta}
			<SyntaxHighlightedCode
				code={meta.code}
				language={FileTypeText.JAVASCRIPT}
				maxHeight={MAX_HEIGHT_CODE_BLOCK}
				streaming={ctx.isCodeStreaming}
			/>
			<div class="mb-2 mt-3 flex items-center gap-2 text-xs text-muted-foreground/70">
				<Terminal class="h-3 w-3" />
				<span>Console</span>
				{#if meta.timeoutMs != null}
					<span class="font-mono">&middot;&nbsp;timeout&nbsp;{meta.timeoutMs}&nbsp;ms</span>
				{/if}
			</div>
			{#if section.toolResult}
				<div class="mt-1">
					<SyntaxHighlightedCode
						code={section.toolResult}
						language={FileTypeText.JAVASCRIPT}
						maxHeight={MAX_HEIGHT_CODE_BLOCK}
					/>
				</div>
			{:else}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">No output</div>
			{/if}
		{/if}
	{/snippet}
</ToolCallBlock>
