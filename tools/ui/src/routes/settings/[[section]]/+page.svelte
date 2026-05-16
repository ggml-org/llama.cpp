<script lang="ts">
	import { SettingsChat } from '$lib/components/app/settings';
	import { page } from '$app/state';
	import { replaceState } from '$app/navigation';
	import { RouterService } from '$lib/services';
	import { SETTINGS_SECTION_SLUGS } from '$lib/constants';
	import { onMount } from 'svelte';

	onMount(() => {
		if (!page.params.section) {
			replaceState(RouterService.settings(SETTINGS_SECTION_SLUGS.GENERAL), {});
		}
	});
</script>

<SettingsChat initialSection={(page.params as Record<string, string | undefined>).section} />
