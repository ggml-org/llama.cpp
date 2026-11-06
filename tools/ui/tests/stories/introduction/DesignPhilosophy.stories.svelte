<script lang="ts" module>
	import { seedChatStory } from '../chat/messages/fixtures';
	import DesignPhilosophy from './DesignPhilosophy.svx';
	import Editorial from './philosophy/Editorial.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the why next to Design guidelines' how, as an editorial page with figures drawn from the real components; the
	// chat stores are seeded for the chat form in its crystal goblet figure
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Introduction/Design philosophy'
	});
</script>

<Story asChild beforeEach={() => seedChatStory()} name="Design philosophy">
	<Editorial>
		<DesignPhilosophy />
	</Editorial>
</Story>
