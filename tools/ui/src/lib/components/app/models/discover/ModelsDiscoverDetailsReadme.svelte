<script lang="ts">
	import { MarkdownContent } from '$lib/components/app';

	interface Props {
		readme: string | null;
	}

	let { readme }: Props = $props();
</script>

{#if readme}
	<section class="space-y-2 bg-muted/50 p-3 rounded-xl">
		<h2 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">README</h2>
		<MarkdownContent content={readme} allowHtml class="prose-sm max-w-none" />
	</section>
{/if}
