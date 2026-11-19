#!/usr/bin/env node
// Decides whether a package is a version npm does not have yet, for
// .github/workflows/ui-lib-publish.yml and ui-chat-publish.yml. It reads the package.json in the
// folder given as its argument, or this library's own without one, so it runs from anywhere.
//
// The version in package.json is the release: this takes it, asks the registry what it holds,
// and says whether to publish, under which dist-tag, and why not when not. The version already
// being on npm is the case that let a release drift once (the library moved, the version did
// not), so it is reported as a verdict rather than passed over in silence.
//
// It writes its findings to $GITHUB_OUTPUT when that is set, and always exits 0 unless the
// registry could not be read: the workflow decides between warning and failing.

import { readFileSync, appendFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REGISTRY = process.env.NPM_REGISTRY_URL ?? 'https://registry.npmjs.org';

const REPO_ROOT = fileURLToPath(new URL('../../../../', import.meta.url));
const pkgDir = process.argv[2] ? resolve(process.argv[2]) : fileURLToPath(new URL('..', import.meta.url));
const pkgPath = relative(REPO_ROOT, join(pkgDir, 'package.json'));
const pkg = JSON.parse(readFileSync(join(pkgDir, 'package.json'), 'utf8'));
const { name, version } = pkg;

if (!name || !version) {
	console.error(`package.json holds no name or version (got ${name}@${version})`);
	process.exit(1);
}

// @scoped/name goes in the registry path with only the slash escaped
const url = `${REGISTRY}/${name.replace('/', '%2F')}`;
let doc;
try {
	const res = await fetch(url);
	if (res.status === 404) {
		doc = null; // never published
	} else if (!res.ok) {
		console.error(`${url} answered ${res.status}`);
		process.exit(1);
	} else {
		doc = await res.json();
	}
} catch (err) {
	console.error(`${url} could not be read: ${err.message}`);
	process.exit(1);
}

const latest = doc?.['dist-tags']?.latest ?? null;
const publishedHere = Boolean(doc?.versions?.[version]);
// a hyphen marks a prerelease; a plus is build metadata, which is not one
const prerelease = version.split('+')[0].includes('-');

let verdict = 'new';
let shouldPublish = true;
let note = `${name}@${version} is not on npm${latest ? `, which holds ${latest}` : ''}`;

if (publishedHere) {
	verdict = 'published';
	shouldPublish = false;
	note = `${name}@${version} is already on npm. Bump the version in ${pkgPath} to release these changes.`;
} else if (latest && compare(version, latest) < 0) {
	// covers the prerelease of an already-released core: 0.4.0-beta.1 behind 0.4.0
	verdict = 'stale';
	shouldPublish = false;
	note = `${name}@${version} is older than the ${latest} on npm.`;
}

const tag = prerelease ? 'next' : 'latest';

console.log(`package.json : ${name}@${version}`);
console.log(`npm latest   : ${latest ?? 'nothing published yet'}`);
console.log(`that version : ${publishedHere ? 'already on npm' : 'not on npm'}`);
console.log(`verdict      : ${verdict} (publish: ${shouldPublish}, dist-tag: ${tag})`);
console.log(`               ${note}`);

if (process.env.GITHUB_OUTPUT) {
	appendFileSync(
		process.env.GITHUB_OUTPUT,
		[
			`name=${name}`,
			`version=${version}`,
			`latest=${latest ?? ''}`,
			`verdict=${verdict}`,
			`should_publish=${shouldPublish}`,
			`tag=${tag}`,
			`note=${note}`,
			'',
		].join('\n'),
	);
}

/**
 * Order two semver versions: negative when a < b, positive when a > b, 0 when they are equal.
 * Written out rather than pulled in as a dependency for a CI check: prerelease identifiers
 * order before the release they lead up to (0.4.0-beta.1 < 0.4.0), fewer identifiers sort
 * first when one is a prefix of the other, numeric identifiers order before alphanumeric ones,
 * and build metadata is ignored (semver 11.4.x of the specification).
 */
function compare(a, b) {
	const [coreA, preA] = split(a);
	const [coreB, preB] = split(b);
	for (let i = 0; i < 3; i++) {
		if (coreA[i] !== coreB[i]) return coreA[i] < coreB[i] ? -1 : 1;
	}
	if (preA.length === 0 && preB.length === 0) return 0;
	if (preA.length === 0) return 1;
	if (preB.length === 0) return -1;
	for (let i = 0; ; i++) {
		const x = preA[i];
		const y = preB[i];
		if (x === undefined && y === undefined) return 0;
		if (x === undefined) return -1;
		if (y === undefined) return 1;
		if (x === y) continue;
		const numX = /^\d+$/.test(x);
		const numY = /^\d+$/.test(y);
		if (numX && numY) return Number(x) < Number(y) ? -1 : 1;
		if (numX) return -1;
		if (numY) return 1;
		return x < y ? -1 : 1;
	}
}

/** Split a version into its numeric core and its prerelease identifiers, dropping build metadata. */
function split(version) {
	const withoutBuild = version.split('+')[0];
	const dash = withoutBuild.indexOf('-');
	const core = (dash === -1 ? withoutBuild : withoutBuild.slice(0, dash)).split('.').map((n) => Number(n) || 0);
	while (core.length < 3) core.push(0);
	const pre = dash === -1 ? [] : withoutBuild.slice(dash + 1).split('.');
	return [core, pre];
}
