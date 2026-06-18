#!/usr/bin/env node

/**
 * Apply circular mask to pwa-*.png icons.
 * Uses the maskable icon as source (white bg, full logo) to avoid
 * the small-colormap pwa icons looking bad when cropped to a circle.
 *
 * Usage: node scripts/make-icons-circular.js [--padding-pct <0-50>] [--scale-pct <50-100>]
 *
 * - padding-pct: percentage of icon size kept as padding around the circle (default: 25)
 * - scale-pct: scale down the source image before cropping (default: 85)
 *
 * maskable-icon and apple-touch-icon are left untouched.
 */

import sharp from 'sharp';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const STATIC_DIR = path.resolve(__dirname, '..', 'static');

const paddingPct = process.argv.reduce((acc, arg, i, args) => {
	if (arg === '--padding-pct' && args[i + 1]) return parseFloat(args[i + 1]);
	return acc;
}, 0);

// Scale down the source image before cropping to circle
const scalePct = process.argv.reduce((acc, arg, i, args) => {
	if (arg === '--scale-pct' && args[i + 1]) return parseFloat(args[i + 1]);
	return acc;
}, 85); // default 85% - icon fills 85% of the circular area

// Source for circular icons: the maskable icon (white bg, full logo)
const sourceIcon = 'maskable-icon-512x512.png';
const targetIcons = ['pwa-64x64.png', 'pwa-192x192.png', 'pwa-512x512.png'];

// maskable-icon and apple-touch-icon stay square
const untouchedIcons = ['maskable-icon-512x512.png', 'apple-touch-icon-180x180.png'];

async function makeCircle(targetFilename) {
	const targetPath = path.join(STATIC_DIR, targetFilename);
	const sourcePath = path.join(STATIC_DIR, sourceIcon);

	if (!fs.existsSync(sourcePath)) {
		console.log(`⏭️  ${sourceIcon} not found, skipping`);
		return;
	}
	if (!fs.existsSync(targetPath)) {
		console.log(`⏭️  ${targetFilename} not found, skipping`);
		return;
	}

	const metadata = await sharp(targetPath).metadata();
	const size = Math.max(metadata.width, metadata.height);
	const radius = Math.floor((size * (1 - paddingPct / 100)) / 2);
	const center = Math.floor(size / 2);

	// Build circular mask as RGBA buffer: white opaque circle on transparent bg
	const maskBuf = Buffer.alloc(size * size * 4, 0);
	for (let y = 0; y < size; y++) {
		for (let x = 0; x < size; x++) {
			const dx = x - center;
			const dy = y - center;
			const dist = Math.sqrt(dx * dx + dy * dy);
			if (dist < radius) {
				const i = (y * size + x) * 4;
				maskBuf[i] = 255;
				maskBuf[i + 1] = 255;
				maskBuf[i + 2] = 255;
				maskBuf[i + 3] = 255;
			}
		}
	}

	const tmpMask = path.join(STATIC_DIR, '.mask-tmp.png');
	await sharp(maskBuf, {
		raw: { width: size, height: size, channels: 4 }
	})
		.png()
		.toFile(tmpMask);

	// Step 1: Scale source relative to circle diameter (not full icon), composite centered onto white canvas of full size
	const circleDiameter = Math.floor(size * (1 - paddingPct / 100));
	const scaledSize = Math.floor((circleDiameter * scalePct) / 100);
	const offset = Math.floor((size - scaledSize) / 2);

	const scaledBuf = await sharp(sourcePath)
		.resize(scaledSize, scaledSize, {
			fit: 'cover',
			background: { r: 255, g: 255, b: 255, alpha: 1 }
		})
		.ensureAlpha()
		.png()
		.toBuffer();

	// Step 2: Composite scaled image onto white background, then apply circular mask
	const output = await sharp({
		create: {
			width: size,
			height: size,
			channels: 4,
			background: { r: 255, g: 255, b: 255, alpha: 1 }
		}
	})
		.composite([
			{ input: scaledBuf, top: offset, left: offset },
			{ input: tmpMask, top: 0, left: 0, blend: 'dest-in' }
		])
		.png()
		.toBuffer();

	fs.writeFileSync(targetPath, output);
	fs.unlinkSync(tmpMask);

	console.log(
		`✓ ${targetFilename} → circle from ${sourceIcon}, ${paddingPct}% padding (size=${size}, r=${radius}, scale=${scalePct}%, circleDiameter=${circleDiameter})`
	);
}

async function main() {
	console.log(`Circular mask: ${paddingPct}% padding, ${scalePct}% scale, source=${sourceIcon}\n`);
	for (const icon of targetIcons) {
		await makeCircle(icon);
	}

	console.log('\nUnchanged:');
	for (const icon of untouchedIcons) {
		const fp = path.join(STATIC_DIR, icon);
		console.log(`  ${icon} (${fs.existsSync(fp) ? fs.statSync(fp).size + ' bytes' : 'missing'})`);
	}
}

main();
