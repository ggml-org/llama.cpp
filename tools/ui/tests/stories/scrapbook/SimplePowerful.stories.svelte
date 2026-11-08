<script lang="ts" module>
	import Editorial from '../philosophy/Editorial.svelte';
	import Figure from '../philosophy/Figure.svelte';
	import Decided from './simple-powerful/Decided.svelte';
	import Depth from './simple-powerful/Depth.svelte';
	import PlainWords from './simple-powerful/PlainWords.svelte';
	import SameScreen from './simple-powerful/SameScreen.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// proposals for Ideology's "Simple by default, powerful on request" figure, each set as it would sit on the page
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Scrapbook/UI/Simple and powerful'
	});
</script>

<Story asChild name="A · Same screen">
	<Editorial>
		<Figure
			caption="The same first screen. Most people see everyday suggestions and a box to type in; a power user opens the settings beside it."
			n={3}
		>
			<SameScreen />
		</Figure>
	</Editorial>
</Story>

<Story asChild name="B · Three steps deep">
	<Editorial>
		<Figure
			caption="How far in someone goes: just ask, choose in plain words, or set every part. Most people stay on the first step."
			n={3}
		>
			<Depth />
		</Figure>
	</Editorial>
</Story>

<Story asChild name="C · Decided for you">
	<Editorial>
		<Figure
			caption="An everyday question, and what the app decided so it could just work. Each decision is a setting a power user can change."
			n={3}
		>
			<Decided />
		</Figure>
	</Editorial>
</Story>

<Story asChild name="D · Plain words">
	<Editorial>
		<Figure
			caption="The same choices in two vocabularies: what most people want, in their words, and the exact setting behind each."
			n={3}
		>
			<PlainWords />
		</Figure>
	</Editorial>
</Story>
