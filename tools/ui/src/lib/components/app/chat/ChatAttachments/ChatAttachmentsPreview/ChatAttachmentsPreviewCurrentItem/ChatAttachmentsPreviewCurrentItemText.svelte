<script lang="ts">
	import { SyntaxHighlightedCode } from '$lib/components/app';

	interface Props {
		displayTextContent: string | undefined;
		language: string;
	}

	let { displayTextContent, language }: Props = $props();
</script>

{#if displayTextContent}
	<div class="px-4 pb-4">
		<SyntaxHighlightedCode
			class="max-w-4xl"
			code={displayTextContent}
			{language}
			maxHeight="none"
		/>
	</div>
{/if}
