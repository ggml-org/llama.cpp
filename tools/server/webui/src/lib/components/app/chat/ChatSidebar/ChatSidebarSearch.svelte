<script lang="ts">
	import { SearchInput } from '$lib/components/app';
	import { t } from '$lib/i18n';

	interface Props {
		value?: string;
		placeholder?: string;
		onInput?: (value: string) => void;
		class?: string;
	}

	let {
		value = $bindable(''),
		placeholder,
		onInput,
		class: className
	}: Props = $props();
</script>

<SearchInput
	bind:value
	placeholder={placeholder ?? t('chat.sidebar.search.placeholder')}
	{onInput}
	class="mb-4 {className}"
/>
