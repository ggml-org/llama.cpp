<script lang="ts">
	import { SettingsChat } from '$lib/components/app/settings';
	import { page } from '$app/state';
</script>

<SettingsChat initialSection={(page.params as Record<string, string | undefined>).section} />
