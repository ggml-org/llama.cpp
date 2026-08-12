<script lang="ts">
	import { afterNavigate, replaceState } from '$app/navigation';
	import { page } from '$app/state';
	import { SettingsChat } from '$lib/components/app/settings';
	import { SETTINGS_SECTION_SLUGS } from '$lib/constants';
	import { RouterService } from '$lib/services';

	afterNavigate(() => {
		if (!page.params.section) {
			replaceState(RouterService.settings(SETTINGS_SECTION_SLUGS.GENERAL), {});
		}
	});
</script>

<SettingsChat initialSection={(page.params as Record<string, string | undefined>).section} />
