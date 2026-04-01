<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ChatSettings } from '$lib/components/app';
	import type { SettingsSectionTitle } from '$lib/constants';

	let sectionParam = $derived(page.url.searchParams.get('section') as SettingsSectionTitle | null);
</script>

<ChatSettings
	onSave={() => goto('#/')}
	initialSection={sectionParam ?? undefined}
	class="mx-auto h-full md:pl-8"
/>
