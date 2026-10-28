<script lang="ts" module>
	import {
		modelServerProps,
		ReasoningEffort,
		resetPointerEvents,
		seedStores,
		ServerRole
	} from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormActionReasoning from '$lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionReasoning.svelte';
	import { screen } from 'storybook/test';

	const { Story } = defineMeta({
		beforeEach: resetPointerEvents,
		component: ChatFormActionReasoning,
		parameters: {
			layout: 'centered'
		},
		title: 'Chat/ChatScreen/ChatForm/ChatFormActionReasoning'
	});

	function seedEffort(
		reasoningEffort: ReasoningEffort,
		serverProps: ApiLlamaCppServerProps = modelServerProps()
	) {
		return () => seedStores({ reasoningEffort, serverProps, serverRole: ServerRole.MODEL });
	}

	const templateLevelsProps = modelServerProps({
		chat_template_caps: { reasoning_effort_levels: ['low', 'medium', 'high', 'xhigh'] }
	});
</script>

<Story beforeEach={seedEffort(ReasoningEffort.DEFAULT)} name="Default" />

<Story beforeEach={seedEffort(ReasoningEffort.OFF)} name="Off" />

<Story beforeEach={seedEffort(ReasoningEffort.HIGH)} name="High" />

<Story
	beforeEach={seedEffort(ReasoningEffort.MEDIUM)}
	name="MenuOpen"
	play={async ({ canvas, userEvent }) => {
		await userEvent.click(canvas.getByRole('button', { name: 'Reasoning effort' }));
		await screen.findByRole('menu');
	}}
/>

<Story
	beforeEach={seedEffort(ReasoningEffort.XHIGH, templateLevelsProps)}
	name="TemplateLevelsMenuOpen"
	play={async ({ canvas, userEvent }) => {
		await userEvent.click(canvas.getByRole('button', { name: 'Reasoning effort' }));
		await screen.findByRole('menu');
	}}
/>
