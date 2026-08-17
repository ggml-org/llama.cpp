<script lang="ts">
	// Typed metadata supplies labels; server content stays text.

	import { FileCode2, FileText } from '@lucide/svelte';
	import ToolCallBlock from './ToolCallBlock.svelte';
	import { classifySkillResourceFormat } from '$lib/components/app/skills/skill-resource-presentation';
	import SkillProviderLabel from '$lib/components/app/skills/SkillProviderLabel.svelte';
	import { resolveSkillSectionMeta } from '$lib/services/skills-activation.service';
	import type { AgenticSection } from '$lib/types';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { isStreaming, onToggle, open, section }: Props = $props();

	const meta = $derived(resolveSkillSectionMeta(section));
	const title = $derived(
		meta
			? meta.kind === 'resource'
				? `Skill resource · ${meta.name}`
				: `Skill · ${meta.name}`
			: 'Skill result'
	);
	const icon = $derived(
		meta?.kind === 'resource' && meta.path
			? classifySkillResourceFormat(meta.path) === 'source'
				? FileCode2
				: FileText
			: undefined
	);
</script>

<ToolCallBlock {section} {open} {isStreaming} meta={null} {title} {icon} {onToggle}>
	{#snippet children(_meta, ctx)}
		{#if ctx.isPending}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Waiting for result...
			</div>
		{:else if section.toolResult}
			{#if meta}
				<div class="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70">
					<SkillProviderLabel provider={meta.provider} />
					<span>· {meta.scope}</span>
					{#if meta.path}<span>· {meta.path}</span>{/if}
				</div>
			{/if}
			<div class="overflow-auto">
				<div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
					{section.toolResult}
				</div>
			</div>
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">No output</div>
		{/if}
	{/snippet}
</ToolCallBlock>
