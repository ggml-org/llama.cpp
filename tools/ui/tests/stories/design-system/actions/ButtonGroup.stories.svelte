<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { ChevronDown, Copy, Pencil, Trash2 } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Button } from '$lib/components/ui/button';
	import * as ButtonGroup from '$lib/components/ui/button-group';

	const { Story } = defineMeta({
		component: ButtonGroup.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Actions/ButtonGroup'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2} minItemWidth="16rem">
		<StateGalleryItem label="Outline">
			<ButtonGroup.Root>
				<Button variant="outline">Copy</Button>

				<Button variant="outline">Edit</Button>

				<Button variant="outline">Delete</Button>
			</ButtonGroup.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Icons">
			<ButtonGroup.Root>
				<Button aria-label="Copy" size="icon" variant="outline"><Copy /></Button>

				<Button aria-label="Edit" size="icon" variant="outline"><Pencil /></Button>

				<Button aria-label="Delete" size="icon" variant="outline"><Trash2 /></Button>
			</ButtonGroup.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Split" note="with separator">
			<ButtonGroup.Root>
				<Button variant="secondary">Send</Button>

				<ButtonGroup.Separator />

				<Button aria-label="More options" size="icon" variant="secondary"><ChevronDown /></Button>
			</ButtonGroup.Root>
		</StateGalleryItem>
	</StateGallery>
</Story>
