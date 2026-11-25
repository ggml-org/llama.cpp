#!/usr/bin/env node
// Turns this checkout of the library into its unscoped alias, llama-ui, for
// .github/workflows/ui-lib-publish.yml: the same build, published a second time under the
// name people type first. Only the name in package.json changes, and the readme gains a note
// saying which package is the real one. Run it after the scoped publish and never commit the
// result.

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const ALIAS = 'llama-ui';

const pkgPath = fileURLToPath(new URL('../package.json', import.meta.url));
const readmePath = fileURLToPath(new URL('../README.md', import.meta.url));

const raw = readFileSync(pkgPath, 'utf8');
const pkg = JSON.parse(raw);
const scoped = pkg.name;

if (scoped === ALIAS) {
	console.log(`package.json is already named ${ALIAS}`);
	process.exit(0);
}
if (!scoped?.endsWith(`/${ALIAS}`)) {
	console.error(`expected a scoped ${ALIAS} package to alias, got ${scoped}`);
	process.exit(1);
}

pkg.name = ALIAS;
// keep the file's own indentation so the diff is the name alone
const indent = raw.match(/^[ \t]+/m)?.[0] ?? '\t';
writeFileSync(pkgPath, JSON.stringify(pkg, null, indent) + '\n');

const note = [
	`> \`${ALIAS}\` is an alias of [\`${scoped}\`](https://www.npmjs.com/package/${scoped}): the same`,
	`> build at the same version, published under both names. Prefer \`${scoped}\`; the examples`,
	`> below use it, and work with \`${ALIAS}\` in its place.`,
	'',
	'',
].join('\n');
writeFileSync(readmePath, note + readFileSync(readmePath, 'utf8'));

console.log(`${scoped}@${pkg.version} is now named ${ALIAS}`);
