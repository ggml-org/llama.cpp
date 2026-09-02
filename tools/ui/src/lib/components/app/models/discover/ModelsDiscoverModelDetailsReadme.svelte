<script lang="ts">
	import { MarkdownContent } from '$lib/components/app';

	interface Props {
		readme: string | null;
	}

	let { readme }: Props = $props();
</script>

{#if readme}
	<section class="bg-muted/50 p-3 rounded-xl">
		<MarkdownContent allowHtml class="prose-sm max-w-none" content={readme} />
	</section>
{/if}
