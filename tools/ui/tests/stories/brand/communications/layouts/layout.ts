// What every key visual takes: its format and ground, the grid overlay, and the top and bottom rows.
import type { FormatKey, Ground } from '../posts/post';

export interface Header {
	brand: 'icon' | 'Llama' | 'llama.cpp';
	kicker?: string;
	pill?: string;
	url?: string;
}

export interface LayoutProps {
	footer?: [string, string];
	format: FormatKey;
	header: Header;
	overlay?: boolean;
	tone: Ground;
}
