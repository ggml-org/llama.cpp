<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { SettingsChat } from '$lib/components/app/settings';
	import type { SettingsSectionTitle } from '$lib/constants';

	let sectionParam = $derived(page.url.searchParams.get('section') as SettingsSectionTitle | null);
</script>

<SettingsChat
	onSave={() => goto('#/')}
	initialSection={sectionParam ?? undefined}
	class="mx-auto max-h-[100dvh] md:pl-8"
/>
