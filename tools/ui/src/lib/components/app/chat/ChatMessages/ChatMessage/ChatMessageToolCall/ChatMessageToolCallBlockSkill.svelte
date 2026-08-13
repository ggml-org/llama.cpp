<script lang="ts">
	// Purpose-built renderer for Skills `read_skill` results (base activations
	// and resource reads). Triggered only when the persisted tool result
	// carries valid typed SKILL metadata; malformed or unknown historical
	// records fall back to the generic tool card. Labels come exclusively
	// from the resolved metadata - the server XML stays opaque text content
	// and is never rendered as UI markup.

	import ToolCallBlock from './ToolCallBlock.svelte';
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
		meta ? (meta.kind === 'resource' ? `Skill resource · ${meta.name}` : `Skill · ${meta.name}`) : 'Skill result'
	);
	const detail = $derived(meta ? [meta.provider, meta.scope, meta.path].filter(Boolean).join(' · ') : '');
</script>

<ToolCallBlock {section} {open} {isStreaming} meta={null} {title} {onToggle}>
	{#snippet children(_meta, ctx)}
		{#if ctx.isPending}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Waiting for result...
			</div>
		{:else if section.toolResult}
			{#if detail}
				<div class="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70">
					{detail}
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
