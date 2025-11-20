<script module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ChatSettings } from '$lib/components/app';
	import { fn } from 'storybook/test';

	const { Story } = defineMeta({
		title: 'Components/ChatSettings',
		component: ChatSettings,
		parameters: {
			layout: 'fullscreen'
		},
		args: {
			onClose: fn(),
			onSave: fn()
		}
	});
</script>

<Story name="Default" />
