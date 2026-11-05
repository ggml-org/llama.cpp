<script lang="ts" module>
	import GlyphButtons from './lockup/GlyphButtons.svelte';
	import LockupContexts from './lockup/LockupContexts.svelte';
	import LockupSteps from './lockup/LockupSteps.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Proposal: the lockup's icon and wordmark proportion, in steps from today's lockup to the icon as a glyph. Delete
	// once it is chosen (see DESIGN.md, "Proposing changes")
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Logo/Lockup proposal'
	});
</script>

<Story asChild name="Steps">
	<LockupSteps />
</Story>

<Story asChild name="In context">
	<LockupContexts />
</Story>

<Story asChild name="Glyph in buttons">
	<GlyphButtons />
</Story>
