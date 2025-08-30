import { useSidebar } from '$lib/components/ui/sidebar';

const sidebar = useSidebar();

export function handleMobileSidebarItemClick() {
	if (sidebar.isMobile) {
		sidebar.toggle();
	}
}
