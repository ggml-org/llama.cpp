#!/usr/bin/env node
// Post-build: copy hashed/nested assets to predictable flat names.
// No file content is modified — the C++ server handles routing hashed URLs
// to the correct stored asset at runtime.
//
// Copies:
//   _app/immutable/bundle.HASH.js         -> bundle.js
//   _app/immutable/assets/bundle.HASH.css -> bundle.css
//   workbox-HEXHASH.js                    -> workbox.js
//   _app/version.json                     -> version.json

import fs from 'fs';
import path from 'path';

const outDir = process.env.LLAMA_UI_OUT_DIR ?? './dist';

function findOne(dir, pattern) {
	const files = fs.readdirSync(dir).filter((f) => pattern.test(f));
	if (files.length === 0) throw new Error(`post-build: no file matching ${pattern} in ${dir}`);
	return path.join(dir, files[0]);
}

function copyFlat(src, destName) {
	const dest = path.join(outDir, destName);
	fs.copyFileSync(src, dest);
	console.log(`post-build: ${path.relative(outDir, src)} -> ${destName}`);
}

const bundleJs = findOne(path.join(outDir, '_app/immutable'), /^bundle\.[^.]+\.js$/);
const bundleCss = findOne(path.join(outDir, '_app/immutable/assets'), /^bundle\.[^.]+\.css$/);
const workbox = findOne(outDir, /^workbox-[0-9a-f]+\.js$/);

copyFlat(bundleJs, 'bundle.js');
copyFlat(bundleCss, 'bundle.css');
copyFlat(workbox, 'workbox.js');

const versionSrc = path.join(outDir, '_app/version.json');
if (fs.existsSync(versionSrc)) {
	copyFlat(versionSrc, 'version.json');
}
