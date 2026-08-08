<script lang="ts">
	import MarkdownContent from '$lib/components/app/content/MarkdownContent/MarkdownContent.svelte';
	import { markdownState } from './markdown-stream-state.svelte';
</script>

<div data-testid="markdown-host">
	<MarkdownContent content={markdownState.content} />
</div>
