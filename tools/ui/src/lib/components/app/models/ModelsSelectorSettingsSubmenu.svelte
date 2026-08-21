<script lang="ts">
	import ModelsSelectorSettingsFields from './ModelsSelectorSettingsFields.svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import type { SettingsSection } from '$lib/types';

	interface Props {
		section: SettingsSection;
		disabled?: boolean;
	}

	let { disabled = false, section }: Props = $props();
</script>

<DropdownMenu.Sub>
	<DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2" {disabled}>
		<section.icon class="h-4 w-4 shrink-0 text-muted-foreground" />

		<span class="text-sm text-muted-foreground">{section.title}</span>
	</DropdownMenu.SubTrigger>

	<DropdownMenu.SubContent
		class="max-h-96 w-72 overflow-y-auto bg-popover p-2 text-popover-foreground shadow-md outline-none"
	>
		<ModelsSelectorSettingsFields {section} />
	</DropdownMenu.SubContent>
</DropdownMenu.Sub>
