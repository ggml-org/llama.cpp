<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import ActionIconsStoryHost from './ActionIconsStoryHost.svelte';
	import { siblingInfo } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		args: {
			actionsPosition: 'left',
			justify: 'start'
		},
		component: ActionIconsStoryHost,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatMessages/ChatMessageActionIcons'
	});

	const onRegenerate = () => {};
</script>

<Story asChild name="States">
	<StateGallery columns={3}>
		<StateGalleryItem label="Assistant" note="regenerate + continue">
			<ActionIconsStoryHost
				actionsPosition="left"
				justify="start"
				onContinue={() => {}}
				{onRegenerate}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="User" note="right aligned">
			<ActionIconsStoryHost actionsPosition="right" justify="end" />
		</StateGalleryItem>

		<StateGalleryItem label="No fork">
			<ActionIconsStoryHost actionsPosition="left" forkable={false} justify="start" />
		</StateGalleryItem>

		<StateGalleryItem label="Raw output switch">
			<ActionIconsStoryHost
				actionsPosition="left"
				justify="start"
				{onRegenerate}
				showRawOutputSwitch
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Branching">
			<ActionIconsStoryHost actionsPosition="left" justify="start" {onRegenerate} {siblingInfo} />
		</StateGalleryItem>

		<StateGalleryItem label="Branching, first version">
			<ActionIconsStoryHost
				actionsPosition="left"
				justify="start"
				{onRegenerate}
				siblingInfo={{ ...siblingInfo, currentIndex: 0 }}
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>

<!-- Separate: the delete dialog opens in a portal over the whole page. -->
<Story
	args={{
		deletionInfo: {
			assistantMessages: 2,
			messageTypes: ['user', 'assistant'],
			totalCount: 4,
			userMessages: 2
		},
		showDeleteDialog: true
	}}
	name="DeleteDialog"
/>
