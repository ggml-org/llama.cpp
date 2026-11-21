// Open state of the context panel, shared by the gauge's popover and the phone's drawer: the
// phone's corner key and the drawer live in different DOM subtrees.
export const gaugePopup = $state({ open: false });

export function gaugePopupToggle(): void {
	gaugePopup.open = !gaugePopup.open;
}

export function gaugePopupClose(): void {
	gaugePopup.open = false;
}
