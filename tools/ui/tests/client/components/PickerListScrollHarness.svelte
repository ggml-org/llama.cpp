<script lang="ts">
	import { ChatFormPickerList, ChatFormPickerListItem } from '$lib/components/app/chat';

	interface Item {
		id: string;
		label: string;
	}

	const items: Item[] = Array.from({ length: 20 }, (_, i) => ({
		id: String(i),
		label: `item ${i}`
	}));

	let open = $state(false);
	let scrollTrigger = $state(0);
	let selectedIndex = $state(0);

	export function openPicker() {
		open = true;
	}
</script>

<div style="height: 5000px;">conversation</div>

{#if open}
	<div data-testid="picker-host">
		<ChatFormPickerList
			isLoading={false}
			itemKey={(it) => it.id}
			{items}
			{scrollTrigger}
			searchQuery=""
			{selectedIndex}
			showSearchInput={false}
		>
			{#snippet item(it, index, isSelected)}
				<ChatFormPickerListItem dataIndex={index} {isSelected} onclick={() => {}}>
					{it.label}
				</ChatFormPickerListItem>
			{/snippet}
		</ChatFormPickerList>
	</div>
{/if}
