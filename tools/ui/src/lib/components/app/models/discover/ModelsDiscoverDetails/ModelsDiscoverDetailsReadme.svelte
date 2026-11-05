<script lang="ts">
	import { MarkdownContent } from '$lib/components/app';

	interface Props {
		readme: string | null;
	}

	let { readme }: Props = $props();
</script>

{#if readme}
	<section class="p-3 rounded-3xl bg-muted/60 elevated-control">
		<MarkdownContent allowHtml class="prose-sm max-w-none" content={readme} />
	</section>
{/if}
