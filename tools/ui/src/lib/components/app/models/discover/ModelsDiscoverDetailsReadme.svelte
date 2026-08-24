<script lang="ts">
	import { MarkdownContent } from '$lib/components/app';

	interface Props {
		readme: string | null;
	}

	let { readme }: Props = $props();
</script>

{#if readme}
	<section class="space-y-2">
		<h2 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">README</h2>
		<article class="rounded-lg border bg-card p-4">
			<MarkdownContent content={readme} class="prose-sm max-w-none" />
		</article>
	</section>
{/if}
