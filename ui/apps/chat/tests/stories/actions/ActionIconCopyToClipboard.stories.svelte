<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ActionIconCopyToClipboard } from '#lib/components/app/index.ts';

	// Llama UI's ActionIcon that copies a text, under messages, code blocks and commands
	const { Story } = defineMeta({
		component: ActionIconCopyToClipboard,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Actions/ActionIconCopyToClipboard'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2}>
		<StateGalleryItem label="Copy to clipboard" note="a green check for 1.5s after copying">
			<ActionIconCopyToClipboard text="llama-server -m model.gguf" />
		</StateGalleryItem>
	</StateGallery>
</Story>
