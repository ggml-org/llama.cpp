<script lang="ts" module>
	import { AI_TUTORIAL_MD } from '../fixtures/ai-tutorial.js';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { MarkdownContent } from '$lib/components/app';

	const { Story } = defineMeta({
		component: MarkdownContent,
		parameters: {
			layout: 'centered'
		},
		title: 'Content/MarkdownContent'
	});
</script>

<Story
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]', content: AI_TUTORIAL_MD }}
	name="Default"
/>
