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
				<Button variant="outline">Send</Button>

				<ButtonGroup.Separator />

				<Button aria-label="More options" size="icon" variant="outline"><ChevronDown /></Button>
			</ButtonGroup.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Small split" note="as in tool permission requests">
			<ButtonGroup.Root>
				<Button size="sm" variant="outline">Allow once</Button>

				<ButtonGroup.Separator />

				<Button aria-label="More allow options" class="px-2" size="sm" variant="outline">
					<ChevronDown />
				</Button>
			</ButtonGroup.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Primary split">
			<ButtonGroup.Root>
				<Button>Send</Button>

				<ButtonGroup.Separator />

				<Button aria-label="More options" size="icon"><ChevronDown /></Button>
			</ButtonGroup.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Primary">
			<ButtonGroup.Root>
				<Button>Copy</Button>

				<Button>Edit</Button>

				<Button>Delete</Button>
			</ButtonGroup.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Secondary">
			<ButtonGroup.Root>
				<Button variant="secondary">Copy</Button>

				<Button variant="secondary">Edit</Button>

				<Button variant="secondary">Delete</Button>
			</ButtonGroup.Root>
		</StateGalleryItem>
	</StateGallery>
</Story>
