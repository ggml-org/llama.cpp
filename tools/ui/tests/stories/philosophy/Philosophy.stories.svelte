<script lang="ts" module>
	import { seedChatStory } from '../chat/messages/fixtures';
	import Design from './Design.svx';
	import Editorial from './Editorial.svelte';
	import Ideology from './Ideology.svx';
	import Product from './Product.svx';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// The why, as editorial pages with figures drawn from the real components: what we stand for, how the products fit
	// together, and why they look and work the way they do. The chat stores are seeded for the chat form in Design's
	// crystal goblet figure
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Philosophy'
	});
</script>

<Story asChild name="Ideology">
	<Editorial>
		<Ideology />
	</Editorial>
</Story>

<Story asChild name="Product">
	<Editorial>
		<Product />
	</Editorial>
</Story>

<Story asChild beforeEach={() => seedChatStory()} name="Design">
	<Editorial>
		<Design />
	</Editorial>
</Story>
