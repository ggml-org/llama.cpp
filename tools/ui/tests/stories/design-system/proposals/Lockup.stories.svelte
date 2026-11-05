<script lang="ts" module>
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

<Story asChild name="Lockup proposal">
	<LockupSteps />
</Story>
