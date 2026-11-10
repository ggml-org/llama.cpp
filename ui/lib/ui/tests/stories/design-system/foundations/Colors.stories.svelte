<script lang="ts" module>
	import { componentGroups, primitiveGroups, semanticGroups } from './tokens';
	import TokenTier from './TokenTier.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// read from src/lib/styles/tokens/*.css, so the swatches follow the CSS as it changes
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'UI/Foundations/Colors'
	});
</script>

<Story asChild name="Primitives">
	<TokenTier
		groups={primitiveGroups}
		intro="Raw values named by what they are. Only semantic tokens point at these; components never use them directly."
		showReference={false}
	/>
</Story>

<Story asChild name="Semantic">
	<TokenTier
		groups={semanticGroups}
		intro="Named by purpose. Each points at a primitive, and this is the only tier that changes between light and dark. Switch the theme to see the dark values."
		showReference={true}
	/>
</Story>

<Story asChild name="Components">
	<TokenTier
		groups={componentGroups}
		intro="One group per component. They point at semantic tokens, so both themes follow automatically."
		showReference={true}
	/>
</Story>
