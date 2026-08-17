<script lang="ts">
	import ChatFormPickers from '$lib/components/app/chat/ChatForm/ChatFormPickers/ChatFormPickers.svelte';
	import type { SkillCatalogEntry } from '$lib/types';

	// Controlled Skills picker harness; owns only the prop boundary.

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
