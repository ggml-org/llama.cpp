// The swag prints, fabrics and placements, as the Swag Creator (ggml-org/swag-creator, src/lib/prints.ts and
// garments.ts) defines them: every print but the quantized L comes from llama-brand, and placements are measured as a
// print shop does, in centimetres from the centre front of the neckline.
import iconDark from '../../assets/files/logo/svg/icon/icon-dark.svg?url';
import iconLight from '../../assets/files/logo/svg/icon/icon-light.svg?url';
import llamaDark from '../../assets/files/logo/svg/lockup/llama/logo-llama-dark.svg?url';
import llamaLight from '../../assets/files/logo/svg/lockup/llama/logo-llama-light.svg?url';
import llamaCppDark from '../../assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-dark.svg?url';
import llamaCppLight from '../../assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-light.svg?url';
import type { Tone } from './quantized';

export interface Artwork {
	id: string;
	label: string;
	/** the placements it may be printed at; any when omitted */
	placements?: string[];
	/** scales the placement's print area for this artwork */
	printScale?: { height: number; width: number };
	/** a llama-brand file per tone; the quantized L is drawn in code instead */
	src?: Record<Tone, string>;
}

export const ARTWORKS: Artwork[] = [
	{ id: 'icon', label: 'Llama icon', src: { dark: iconDark, light: iconLight } },
	{ id: 'llama-logo', label: 'Llama logo', src: { dark: llamaDark, light: llamaLight } },
	{
		id: 'llama-cpp-logo',
		label: 'llama.cpp logo',
		src: { dark: llamaCppDark, light: llamaCppLight }
	},
	// its detail only reads at the large centre size; the legend makes it 1.5 times wider than tall
	{
		id: 'quantized',
		label: 'L · Quantized',
		placements: ['center'],
		printScale: { height: 0.8, width: 1.2 }
	}
];

export interface Fabric {
	id: string;
	label: string;
	/** how the fabric reads, for a flat swatch */
	swatch: string;
	/** which print variant reads on it: `light` has dark ink */
	tone: Tone;
}

export const FABRICS: Fabric[] = [
	{ id: 'white', label: 'White', swatch: '#f5f5f3', tone: 'light' },
	{ id: 'melange', label: 'Melange', swatch: '#bdbdbb', tone: 'light' },
	{ id: 'black', label: 'Black', swatch: '#1c1c1c', tone: 'dark' }
];

export interface Placement {
	height: number;
	id: string;
	label: string;
	/** distance from the neckline down to the print's top edge */
	top: number;
	width: number;
	/** the print centre's offset; positive is the wearer's left, the viewer's right */
	x: number;
}

export const PLACEMENTS: Placement[] = [
	{ height: 5, id: 'collar', label: 'Below collar', top: 5, width: 8, x: 0 },
	{ height: 2.5, id: 'left-chest', label: 'Left chest', top: 10, width: 5, x: 11.5 },
	{ height: 17, id: 'center', label: 'Center', top: 12, width: 17, x: 0 }
];
