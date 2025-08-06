<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatSettingsSection from '$lib/components/app/chat/ChatSettings/ChatSettingsSection.svelte';
	import { Settings, Filter, Beaker } from '@lucide/svelte';

	const { Story } = defineMeta({
		title: 'Components/ChatSettings/ChatSettingsSection',
		component: ChatSettingsSection,
		parameters: {
			layout: 'centered'
		}
	});
</script>

<Story
	name="General"
	args={{
		title: 'General',
		icon: Settings,
		class: 'w-80'
	}}
>
	<span>something</span>
</Story>
