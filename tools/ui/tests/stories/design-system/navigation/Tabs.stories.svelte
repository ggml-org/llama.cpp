<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { Cpu, MessageSquare, Wrench } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import * as Tabs from '$lib/components/ui/tabs';

	const { Story } = defineMeta({
		component: Tabs.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Navigation/Tabs'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2} minItemWidth="20rem">
		<StateGalleryItem label="Default" note="segmented track">
			<Tabs.Root value="general">
				<Tabs.List>
					<Tabs.Trigger value="general">General</Tabs.Trigger>

					<Tabs.Trigger value="sampling">Sampling</Tabs.Trigger>

					<Tabs.Trigger value="tools">Tools</Tabs.Trigger>
				</Tabs.List>

				<Tabs.Content class="pt-2 text-fg-muted" value="general">General settings.</Tabs.Content>

				<Tabs.Content class="pt-2 text-fg-muted" value="sampling">Sampling settings.</Tabs.Content>

				<Tabs.Content class="pt-2 text-fg-muted" value="tools">Tool settings.</Tabs.Content>
			</Tabs.Root>
		</StateGalleryItem>

		<StateGalleryItem label="With icons">
			<Tabs.Root value="chat">
				<Tabs.List>
					<Tabs.Trigger value="chat"><MessageSquare /> Chat</Tabs.Trigger>

					<Tabs.Trigger value="model"><Cpu /> Model</Tabs.Trigger>

					<Tabs.Trigger value="tools"><Wrench /> Tools</Tabs.Trigger>
				</Tabs.List>
			</Tabs.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Line">
			<Tabs.Root value="general">
				<Tabs.List variant="line">
					<Tabs.Trigger value="general">General</Tabs.Trigger>

					<Tabs.Trigger value="sampling">Sampling</Tabs.Trigger>

					<Tabs.Trigger value="tools">Tools</Tabs.Trigger>
				</Tabs.List>
			</Tabs.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Disabled tab">
			<Tabs.Root value="general">
				<Tabs.List>
					<Tabs.Trigger value="general">General</Tabs.Trigger>

					<Tabs.Trigger disabled value="sampling">Sampling</Tabs.Trigger>
				</Tabs.List>
			</Tabs.Root>
		</StateGalleryItem>
	</StateGallery>
</Story>
