<script lang="ts">
	import ChatFormPickers from '$lib/components/app/chat/ChatForm/ChatFormPickers/ChatFormPickers.svelte';
	import type { SkillCatalogEntry } from '$lib/types';

	// Composition harness for the `ChatFormPickers` Skills props: binds the
	// picker to a controlled in-memory catalog (`skills`) and records the
	// selected candidate name. The real CWD/ready-slot derivation in
	// `ChatForm.svelte` is exercised by the live browser smoke test; this
	// harness owns the prop boundary only.

	let { skills = [] }: { skills?: SkillCatalogEntry[] } = $props();
	let selectedName = $state<string | null>(null);

	export function getSelectedName() {
		return selectedName;
	}
</script>

<ChatFormPickers
	isSkillPickerOpen
	skillQuery="alpha"
	{skills}
	onSkillPickerClose={() => {}}
	onSkillSelect={(name) => (selectedName = name)}
/>
