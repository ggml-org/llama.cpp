/**
 * The context gauge's dial at each size. The sm and md dials draw on a 32 grid, md filling its
 * box, as the ring would sit inside a padding ring otherwise. The key dial draws on the icon grid
 * instead (24, a 2px stroke), as wide as the menu key's glyph across from it, so the two keys
 * read alike, and its track is stronger to show on the key's surface.
 */
export const CONTEXT_GAUGE_DIAL_SIZES = {
	/** the icon of a phone's corner key */
	key: { grid: 24, radius: 8.5, sizeClass: 'size-6', strokeWidth: 2, trackOpacity: 0.2 },
	/** a phone's inline touch target */
	md: { grid: 32, radius: 13, sizeClass: 'size-6', strokeWidth: 4, trackOpacity: 0.1 },
	/** inline */
	sm: { grid: 32, radius: 11, sizeClass: 'size-5', strokeWidth: 3, trackOpacity: 0.1 }
} as const;
