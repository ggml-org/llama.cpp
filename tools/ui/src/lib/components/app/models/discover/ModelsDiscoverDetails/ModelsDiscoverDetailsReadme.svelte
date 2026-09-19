<script lang="ts">
	import { MarkdownContent } from '$lib/components/app';

	interface Props {
		readme: string | null;
	}

	let { readme }: Props = $props();
</script>

{#if readme}
	<section
		class="p-3 rounded-3xl border border-border/30 bg-muted/60 shadow-xs dark:border-border/20"
	>
		<MarkdownContent allowHtml class="prose-sm max-w-none" content={readme} />
	</section>
{/if}
