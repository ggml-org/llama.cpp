<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { Pencil, RefreshCw, Trash2 } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ActionIcon } from '@ggml-org/llama-ui/app/actions';

	// the small icon buttons under messages, code blocks and list rows
	const { Story } = defineMeta({
		component: ActionIcon,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/App components/ActionIcon'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2}>
		<StateGalleryItem label="Ghost" note="default">
			<div class="flex items-center gap-1">
				<ActionIcon ariaLabel="Edit" icon={Pencil} tooltip="Edit" />

				<ActionIcon ariaLabel="Regenerate" icon={RefreshCw} tooltip="Regenerate" />

				<ActionIcon ariaLabel="Delete" icon={Trash2} tooltip="Delete" />
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<ActionIcon ariaLabel="Edit" disabled icon={Pencil} tooltip="Edit" />
		</StateGalleryItem>
	</StateGallery>
</Story>
