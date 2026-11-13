<script lang="ts">
	import { SyntaxHighlightedCode } from '$lib/components/app';

	interface Props {
		displayTextContent: string | undefined;
		language: string;
	}

	let { displayTextContent, language }: Props = $props();
</script>

{#if displayTextContent}
	<div class="px-4 pb-4 min-h-0 min-w-0 max-w-full md:max-w-[80vw]">
		<SyntaxHighlightedCode code={displayTextContent} {language} maxHeight="80dvh" />
	</div>
{/if}
