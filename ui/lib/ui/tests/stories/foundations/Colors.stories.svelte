<script lang="ts" module>
	import { componentGroups, primitiveGroups, semanticGroups } from './tokens';
	import TokenTier from './TokenTier.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// read from ui/lib/ui/src/lib/styles/tokens/*.css, so the swatches follow the CSS as it changes
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Foundations/Colors'
	});
</script>

<Story asChild name="Primitives">
	<TokenTier
		groups={primitiveGroups}
		intro="Raw colours named by what they are. Only semantic tokens point at these; components never use them directly."
		showReference={false}
	/>
</Story>

<Story asChild name="Semantic">
	<TokenTier
		groups={semanticGroups}
		intro="Named by purpose. Each picks a primitive per theme or is mixed from another semantic token by a fixed rule, and this is the only tier that changes between light and dark. Switch the theme to see the dark values."
		showReference={true}
	/>
</Story>

<Story asChild name="Components">
	<TokenTier
		groups={componentGroups}
		intro="Only what a component decides itself: a value no semantic token has, or one that only this component changes between themes."
		showReference={true}
	/>
</Story>
