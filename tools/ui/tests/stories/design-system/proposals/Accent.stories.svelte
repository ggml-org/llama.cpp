<script lang="ts" module>
	import AccentProposal from './accent/AccentProposal.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Proposal: one accent orange for the brand, judged on the real gallery and key visuals. Delete once it is chosen
	// (see DESIGN.md, "Proposing changes")
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Colors/Accent proposal'
	});
</script>

<Story asChild name="Accent proposal">
	<AccentProposal />
</Story>
