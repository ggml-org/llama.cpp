<script lang="ts" module>
	import {
		cwdServerTools,
		fileAttachments,
		FORM_CLASS,
		globSearchResult,
		jpgAsset,
		loadingAttachment,
		mcpResources,
		mcpServers,
		mentionEntries,
		modelServerProps,
		ReasoningEffort,
		resetPointerEvents,
		routerModels,
		seedStores,
		SERVER_HOME,
		ServerRole,
		svgAsset,
		toGlobEntries
	} from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreenForm from '$lib/components/app/chat/ChatScreen/ChatScreenForm.svelte';
	import { ATTACHMENT_TOOLTIP_TEXT } from '$lib/constants';
	import { expect, screen } from 'storybook/test';

	const { Story } = defineMeta({
		beforeEach: resetPointerEvents,
		component: ChatScreenForm,
		parameters: {
			layout: 'centered'
		},
		title: 'Chat/ChatScreen/ChatForm/ChatScreenForm'
	});

	const modelServer = {
		serverProps: modelServerProps(),
		serverRole: ServerRole.MODEL
	};
</script>

<Story
	args={{ class: FORM_CLASS }}
	name="Default"
	play={async ({ canvas, userEvent }) => {
		const textarea = await canvas.findByRole('textbox');
		const submitButton = await canvas.findByRole('button', { name: 'Send' });

		// Expect the input to be focused after the component is mounted
		await expect(textarea).toHaveFocus();

		// Expect the submit button to be disabled
		await expect(submitButton).toBeDisabled();

		const text = 'What is the meaning of life?';

		await userEvent.clear(textarea);
		await userEvent.type(textarea, text);

		await expect(textarea).toHaveValue(text);

		const fileInput = document.querySelector('input[type="file"]');

		await expect(fileInput).not.toHaveAttribute('accept');
	}}
/>

<Story
	args={{ class: FORM_CLASS }}
	beforeEach={() => seedStores(modelServer)}
	name="WithText"
	play={async ({ canvas, userEvent }) => {
		const textarea = await canvas.findByRole('textbox');

		await userEvent.type(textarea, 'Explain how the KV cache works in llama.cpp');
	}}
/>

<Story
	args={{ class: FORM_CLASS }}
	beforeEach={() => seedStores(modelServer)}
	name="WithRichInput"
	play={async ({ canvas, userEvent }) => {
		const textarea = await canvas.findByRole('textbox');

		// a complete code span swaps the textarea for the rich input
		await userEvent.type(textarea, 'Why does `npm run check` fail after `npm install`?');
	}}
/>

<Story args={{ class: FORM_CLASS, isLoading: true }} name="Loading" />

<Story args={{ class: FORM_CLASS, disabled: true }} name="Disabled" />

<Story
	args={{
		class: FORM_CLASS,
		uploadedFiles: fileAttachments
	}}
	name="WithFileAttachments"
	play={async ({ canvas }) => {
		const jpgAttachment = canvas.getByAltText('1.jpg');
		const svgAttachment = canvas.getByAltText('hf-logo.svg');

		await expect(jpgAttachment).toHaveAttribute('src', jpgAsset);
		await expect(svgAttachment).toHaveAttribute('src', svgAsset);
		await expect(canvas.getByText('example.pdf')).toBeInTheDocument();
		await expect(canvas.getByText('342.82 KB')).toBeInTheDocument();
	}}
/>

<Story
	args={{ class: FORM_CLASS, uploadedFiles: [fileAttachments[0], loadingAttachment] }}
	name="WithLoadingAttachment"
/>

<Story
	args={{ class: FORM_CLASS }}
	beforeEach={() =>
		seedStores({
			config: { autoMicOnEmpty: true },
			serverProps: modelServerProps({ modalities: { audio: true, video: false, vision: true } }),
			serverRole: ServerRole.MODEL
		})}
	name="WithRecordButton"
/>

<Story
	args={{ class: FORM_CLASS }}
	beforeEach={() => seedStores({ ...modelServer, reasoningEffort: ReasoningEffort.HIGH })}
	name="WithReasoningEffort"
/>

<Story
	args={{ class: FORM_CLASS }}
	beforeEach={() =>
		seedStores({
			models: routerModels,
			selectedModelId: 'removed-model',
			serverProps: modelServerProps({ role: ServerRole.ROUTER }),
			serverRole: ServerRole.ROUTER
		})}
	name="ModelNotAvailable"
	play={async ({ canvas, userEvent }) => {
		const textarea = await canvas.findByRole('textbox');

		await userEvent.type(textarea, 'Hello');
	}}
/>

<Story
	args={{ class: FORM_CLASS }}
	beforeEach={() => seedStores({ mcpResources, mcpServers })}
	name="WithMcpResources"
/>

<Story
	args={{ class: FORM_CLASS }}
	beforeEach={() =>
		seedStores({
			...modelServer,
			pendingCwd: `${SERVER_HOME}/llama.cpp/tools/ui`,
			serverHome: SERVER_HOME,
			serverTools: cwdServerTools
		})}
	name="WithWorkingDirectory"
/>

<Story
	args={{ class: FORM_CLASS }}
	name="AddMenuOpen"
	play={async ({ canvas, userEvent }) => {
		await userEvent.click(await canvas.findByRole('button', { name: ATTACHMENT_TOOLTIP_TEXT }));
		await screen.findByRole('menu');
	}}
/>

<Story
	args={{ class: FORM_CLASS }}
	beforeEach={() =>
		seedStores({
			...modelServer,
			mcpServers,
			serverHome: SERVER_HOME,
			serverTools: cwdServerTools
		})}
	name="CommandPickerOpen"
	play={async ({ canvas, userEvent }) => {
		await userEvent.type(await canvas.findByRole('textbox'), '/');
		await screen.findByText('/prompt');
	}}
/>

<Story
	args={{ class: FORM_CLASS }}
	beforeEach={() =>
		seedStores({
			...modelServer,
			globSearch: globSearchResult(toGlobEntries(mentionEntries)),
			serverHome: SERVER_HOME,
			serverTools: cwdServerTools
		})}
	name="MentionPickerOpen"
	play={async ({ canvas, userEvent }) => {
		await userEvent.type(await canvas.findByRole('textbox'), 'Look at @chat');
		await screen.findByText('ChatForm.svelte');
	}}
/>
