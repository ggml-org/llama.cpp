<script module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ChatSettingsDialog } from '$lib/components/app';
	import { fn } from 'storybook/test';

	const { Story } = defineMeta({
		title: 'Components/ChatSettingsDialog',
		component: ChatSettingsDialog,
		parameters: {
			layout: 'fullscreen'
		},
		argTypes: {
			open: {
				control: 'boolean',
				description: 'Whether the dialog is open'
			}
		},
		args: {
			onOpenChange: fn()
		}
	});
</script>

<Story name="Open" args={{ open: true }} />

<Story name="Closed" args={{ open: false }} />
