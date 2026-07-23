<script lang="ts">
	import { deriveAgenticSections, buildAssistantRawOutput } from '$lib/utils';

	interface Props {
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
	}

	let { message, toolMessages = [] }: Props = $props();

	let rawOutputContent = $derived.by(() => {
		const sections = deriveAgenticSections(message, toolMessages, [], false);
		return buildAssistantRawOutput(sections);
	});
</script>

<pre class="raw-output">{rawOutputContent || ''}</pre>

<style>
	.raw-output {
		width: 100%;
		max-width: 48rem;
		margin-top: 1.5rem;
		padding: 1rem 1.25rem;
		border-radius: 1rem;
		background: hsl(var(--muted) / 0.3);
		color: var(--foreground);
		font-size: 0.875rem;
		line-height: 1.6;
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>
