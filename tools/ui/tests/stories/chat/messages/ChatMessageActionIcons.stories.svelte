<script lang="ts" module>
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
</script>

<Story args={{ onContinue: () => {}, onRegenerate: () => {} }} name="Default" />

<Story args={{ actionsPosition: 'right', justify: 'end' }} name="User" />

<Story args={{ forkable: false }} name="NoFork" />

<Story args={{ onRegenerate: () => {}, showRawOutputSwitch: true }} name="WithRawOutputSwitch" />

<Story args={{ onRegenerate: () => {}, siblingInfo }} name="WithBranching" />

<Story
	args={{ onRegenerate: () => {}, siblingInfo: { ...siblingInfo, currentIndex: 0 } }}
	name="BranchingFirstVersion"
/>

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
