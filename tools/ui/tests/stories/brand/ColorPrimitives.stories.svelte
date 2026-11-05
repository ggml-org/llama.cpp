<script lang="ts" module>
	import { primitiveGroups } from '../design-system/foundations/tokens';
	import TokenTier from '../design-system/foundations/TokenTier.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the palette's raw values (greys, beige, the status hues, the art colours and cinder), read from
	// src/lib/styles/tokens/primitives.css so the swatches follow the CSS as it changes
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Brand/Colors/Primitives'
	});
</script>

<Story asChild name="Primitives">
	<TokenTier
		groups={primitiveGroups}
		intro="Raw values named by what they are. Only semantic tokens point at these; components never use them directly."
		showReference={false}
	/>
</Story>
