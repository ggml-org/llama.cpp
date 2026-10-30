<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { Pencil, RefreshCw, Trash2 } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ActionIcon, ActionIconCopyToClipboard } from '$lib/components/app';

	// the small icon buttons under messages, code blocks and list rows
	const { Story } = defineMeta({
		component: ActionIcon,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Patterns/ActionIcon'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={3}>
		<StateGalleryItem label="Ghost" note="default">
			<div class="flex items-center gap-1">
				<ActionIcon ariaLabel="Edit" icon={Pencil} tooltip="Edit" />

				<ActionIcon ariaLabel="Regenerate" icon={RefreshCw} tooltip="Regenerate" />

				<ActionIcon ariaLabel="Delete" icon={Trash2} tooltip="Delete" />
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Copy to clipboard" note="a green check for 1.5s after copying">
			<ActionIconCopyToClipboard text="llama-server -m model.gguf" />
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<ActionIcon ariaLabel="Edit" disabled icon={Pencil} tooltip="Edit" />
		</StateGalleryItem>
	</StateGallery>
</Story>
