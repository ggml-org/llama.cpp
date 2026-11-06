<script lang="ts" module>
	import { seedChatStory } from '../chat/messages/fixtures';
	import Design from './Design.svx';
	import Editorial from './Editorial.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Why the brand, Llama UI and Llama Chat look and work the way they do, with figures drawn from the
	// real components; the chat stores are seeded for the chat form in its crystal goblet figure
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Philosophy/Design'
	});
</script>

<Story asChild beforeEach={() => seedChatStory()} name="Design">
	<Editorial>
		<Design />
	</Editorial>
</Story>
