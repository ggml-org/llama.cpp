/**
 * Records demo videos for pull requests and docs. An app runs it from its own demo/cli.ts:
 *
 *   pr <number>   record the ```ts demo block from a PR description, a video per test
 *                 (--trust <sha> to record someone else's PR at that head commit)
 *   docs          record every docs/*.demo.ts script in DEMO_STORAGE
 *
 * plus any commands of the app's own. Settings come from the app's .env.
 */

import { type DemoStorage, getDemoStorage, getPublicUrl } from "./storage.ts";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { basename, join, relative } from "node:path";

/** An app the CLI records demos of. */
export interface DemoApp {
  /** More commands, such as writing the app's fixtures, each given the arguments after it. */
  commands?: Record<string, (args: string[]) => Promise<void> | void>;
  /** The app's demo folder: its helpers.ts, playwright.config.ts and videos/. */
  demoDir: string;
  /** The app's folder, with its node_modules and .env, where Playwright runs. */
  dir: string;
  /**
   * Folders, relative to `dir`, whose node_modules a PR's worktree also needs, such as a
   * library the app builds from source. The app's own are always linked.
   */
  linkNodeModules?: string[];
}

let app: DemoApp;
let REPO = "";
const runsDir = () => join(app.demoDir, ".runs");
const videosDir = () => join(app.demoDir, "videos");
const DEMO_BLOCK = /```(?:ts|typescript)[ \t]+demo[ \t]*\r?\n([\s\S]*?)```/;
const HELPERS_IMPORT =
  /^import\s[^;]*from\s+['"][^'"]*helpers(?:\.ts)?['"];?\s*$/gm;
const SECTION_START = "<!-- demo:start -->";
const SECTION_END = "<!-- demo:end -->";
/** GitHub's image proxy refuses larger images, so a bigger preview shows as a broken image. */
const PREVIEW_MAX_BYTES = 5 * 1024 * 1024;
function run(
  cmd: string,
  args: string[],
  opts: { cwd?: string; env?: NodeJS.ProcessEnv } = {},
): string {
  return execFileSync(cmd, args, {
    cwd: opts.cwd ?? app.dir,
    encoding: "utf8",
    env: opts.env,
  }).trim();
}

/** Writes a demo script into .runs/, importing the harness helpers. */
function writeDemoScript(name: string, source: string): string {
  const file = join(runsDir(), `${name}.demo.ts`);
  const helpers = relative(runsDir(), join(app.demoDir, "helpers.ts"));
  const body = source.replace(HELPERS_IMPORT, "").trim();

  mkdirSync(runsDir(), { recursive: true });
  writeFileSync(
    file,
    `import { expect, test } from './${helpers}';\n\n${body}\n`,
  );

  return file;
}

async function freePort(): Promise<number> {
  return new Promise((done, fail) => {
    const server = createServer();

    server.once("error", fail);
    server.listen(0, () => {
      const { port } = server.address() as { port: number };

      server.close(() => done(port));
    });
  });
}

/** A video recorded from one `test()` of a script. */
interface Video {
  /** From the test's `description` annotation, shown beside its video in the PR. */
  description?: string;
  mp4: string;
  title: string;
}

/** Records the given scripts; returns a video for each of their tests, in order. */
async function record(scripts: string[], uiDir = app.dir): Promise<Video[]> {
  const port = await freePort();
  const manifests = scripts.map((s) =>
    join(videosDir(), `${basename(s, ".demo.ts")}.videos.jsonl`),
  );

  for (const manifest of manifests) rmSync(manifest, { force: true });

  execFileSync(
    "npx",
    [
      "playwright",
      "test",
      `--config=${relative(app.dir, join(app.demoDir, "playwright.config.ts"))}`,
      ...scripts.map((s) => relative(app.dir, s)),
    ],
    {
      cwd: app.dir,
      env: {
        ...process.env,
        DEMO_STORYBOOK_PORT: String(await freePort()),
        DEMO_UI_DIR: uiDir,
        DEMO_UI_PORT: String(port),
      },
      stdio: "inherit",
    },
  );

  // helpers.ts lists each video it encodes in the script's manifest
  return manifests.flatMap((manifest) =>
    readFileSync(manifest, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Video),
  );
}

/**
 * Animated GIF preview that GitHub renders inline, its longer side DEMO_PREVIEW_WIDTH (so a
 * phone's portrait video is that tall), shrunk until it fits under PREVIEW_MAX_BYTES.
 */
function makePreview(mp4: string): string {
  const gif = mp4.replace(/\.mp4$/, ".gif");

  for (
    let side = Number(process.env.DEMO_PREVIEW_WIDTH ?? 800);
    ;
    side = Math.round(side * 0.8)
  ) {
    execFileSync("ffmpeg", [
      "-y",
      "-loglevel",
      "error",
      "-i",
      mp4,
      "-vf",
      `fps=10,scale='if(gt(iw,ih),${side},-2)':'if(gt(iw,ih),-2,${side})':flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle`,
      "-loop",
      "0",
      gif,
    ]);

    if (statSync(gif).size <= PREVIEW_MAX_BYTES || side < 320) return gif;
  }
}

function upload(storage: DemoStorage, files: Record<string, string>): void {
  for (const [key, file] of Object.entries(files)) {
    storage.upload(file, key);
    console.log(`uploaded ${key}`);
  }
}

/**
 * Public URL with a content hash, so a re-recorded file at the same key gets a new
 * URL: GitHub's image proxy caches previews by URL.
 */
function versionedUrl(key: string, file: string): null | string {
  const url = getPublicUrl(key);
  const version = createHash("sha1")
    .update(readFileSync(file))
    .digest("hex")
    .slice(0, 10);

  return url && `${url}?v=${version}`;
}

/** Whether a video is taller than wide, like a phone's. */
function isPortrait(mp4: string): boolean {
  const [width, height] = run("ffprobe", [
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=width,height",
    "-of",
    "csv=p=0",
    mp4,
  ])
    .split(",")
    .map(Number);

  return height > width;
}

/**
 * One preview, or a table of them when the script has several tests, each headed by its
 * test's title: up to four portrait videos to a row, or two landscape ones, so each stays
 * large enough to read. When the tests have descriptions, a video to a row instead, with its
 * title and description beside it.
 */
function demoSection(
  links: { description?: string; gif: string; mp4: string; title: string }[],
  sha: string,
  columns: number,
): string {
  const preview = ({ gif, mp4, title }: (typeof links)[number]) =>
    `[![${title.replace(/[[\]|]/g, "")}](${gif})](${mp4})`;
  // a line break would end the table row, so it becomes one inside the cell
  const cell = (text: string) =>
    text.replace(/\|/g, "\\|").replace(/\r?\n/g, "<br>");
  const rows = Array.from(
    { length: Math.ceil(links.length / columns) },
    (_, i) => links.slice(i * columns, (i + 1) * columns),
  );
  const pad = (cells: string[]) => [
    ...cells,
    ...Array(columns - cells.length).fill(" "),
  ];
  const described = [
    "| Video | What it shows |",
    "|---|---|",
    ...links.map(
      (l) =>
        `| ${preview(l)} | **${cell(l.title)}**${l.description ? `<br><br>${cell(l.description)}` : ""} |`,
    ),
  ];
  const body = links.some((l) => l.description)
    ? described
    : links.length === 1
      ? [preview({ ...links[0], title: "Demo" })]
      : rows.flatMap((row, i) => [
          // a table has one header row, so later rows title their videos in bold
          `| ${pad(row.map((l) => (i === 0 ? cell(l.title) : `**${cell(l.title)}**`))).join(" | ")} |`,
          ...(i === 0 ? [`|${Array(columns).fill("---").join("|")}|`] : []),
          `| ${pad(row.map(preview)).join(" | ")} |`,
        ]);
  const note =
    links.length === 1
      ? "the preview for the full video"
      : "a preview for its full video";

  return [
    SECTION_START,
    ...body,
    "",
    `Recorded at ${sha.slice(0, 10)}; click ${note}.`,
    SECTION_END,
  ].join("\n");
}

/** Replaces the managed demo section, or adds a `## Demo` section before the details. */
function withDemoSection(body: string, section: string): string {
  const start = body.indexOf(SECTION_START);
  const end = body.indexOf(SECTION_END);

  if (start !== -1 && end > start) {
    return (
      body.slice(0, start) + section + body.slice(end + SECTION_END.length)
    );
  }

  const block = `## Demo\n\n${section}\n\n`;
  const anchor = body.search(/^## (Additional information|Requirements)\b/m);

  return anchor === -1
    ? `${body.trimEnd()}\n\n${block}`
    : body.slice(0, anchor) + block + body.slice(anchor);
}

async function recordPr(number: string, trustedSha?: string): Promise<void> {
  const pr = JSON.parse(
    run("gh", [
      "pr",
      "view",
      number,
      "--repo",
      REPO,
      "--json",
      "author,body,headRefOid",
    ]),
  ) as { author: { login: string }; body: string; headRefOid: string };
  const source = DEMO_BLOCK.exec(pr.body)?.[1];

  if (!source)
    throw new Error(
      `PR #${number} has no \`\`\`ts demo block in its description`,
    );

  const sha = pr.headRefOid;

  // the PR's code and demo script run on this machine, with the gh and hf logins,
  // so someone else's PR only runs at a head commit you've read and passed to --trust
  if (pr.author.login !== run("gh", ["api", "user", "--jq", ".login"])) {
    if (!trustedSha) {
      throw new Error(
        `PR #${number} is by ${pr.author.login}, and recording it runs its code here: ` +
          `read its diff and demo block, then run again with --trust ${sha.slice(0, 10)}`,
      );
    }

    if (trustedSha.length < 7 || !sha.startsWith(trustedSha)) {
      throw new Error(
        `PR #${number} head is ${sha.slice(0, 10)}, not the trusted ${trustedSha}`,
      );
    }
  }

  const worktree = join(tmpdir(), `llama-ui-demo-pr-${number}`);

  run("git", [
    "fetch",
    "--no-tags",
    `https://github.com/${REPO}.git`,
    `pull/${number}/head`,
  ]);

  if (run("git", ["rev-parse", "FETCH_HEAD"]) !== sha) {
    throw new Error(`PR #${number} head moved while fetching, run again`);
  }

  rmSync(worktree, { force: true, recursive: true });
  run("git", ["worktree", "prune"]);
  run("git", ["worktree", "add", "--detach", worktree, sha]);
  // git's own path from the top to the app, so a symlinked checkout path cannot skew it
  const appInWorktree = join(
    worktree,
    run("git", ["rev-parse", "--show-prefix"], { cwd: app.dir }),
  );

  for (const dir of [".", ...(app.linkNodeModules ?? [])]) {
    const modules = join(app.dir, dir, "node_modules");

    if (existsSync(modules))
      symlinkSync(modules, join(appInWorktree, dir, "node_modules"));
  }

  const name = `pr-${number}`;
  const script = writeDemoScript(name, source);

  try {
    const videos = (await record([script], appInWorktree)).map((video) => ({
      ...video,
      gif: makePreview(video.mp4),
    }));
    const storage = getDemoStorage();

    for (const { gif, mp4 } of videos)
      console.log(`\nvideo:   ${mp4}\npreview: ${gif}`);

    if (!storage) {
      console.log(
        "DEMO_STORAGE is not set: drag the mp4 into the PR description.",
      );

      return;
    }

    const prefix = `videos/pr/${number}/${sha.slice(0, 10)}`;
    // a script with several tests gets <sha>-1.mp4, <sha>-2.mp4 and so on
    const keys = videos.map((_, i) =>
      videos.length === 1 ? prefix : `${prefix}-${i + 1}`,
    );

    upload(storage, { [`${prefix}.demo.ts`]: script });
    videos.forEach(({ gif, mp4 }, i) => {
      upload(storage, { [`${keys[i]}.gif`]: gif, [`${keys[i]}.mp4`]: mp4 });
    });

    if (!getPublicUrl(prefix)) {
      console.log(
        "DEMO_STORAGE_PUBLIC_URL is not set: not linking the video in the PR.",
      );

      return;
    }

    const links = videos.map(({ description, gif, mp4, title }, i) => ({
      description,
      gif: versionedUrl(`${keys[i]}.gif`, gif) ?? "",
      mp4: versionedUrl(`${keys[i]}.mp4`, mp4) ?? "",
      title,
    }));

    execFileSync(
      "gh",
      ["pr", "edit", number, "--repo", REPO, "--body-file", "-"],
      {
        cwd: app.dir,
        input: withDemoSection(
          pr.body,
          demoSection(
            links,
            sha,
            Math.min(links.length, isPortrait(videos[0].mp4) ? 4 : 2),
          ),
        ),
        stdio: ["pipe", "inherit", "inherit"],
      },
    );
    console.log(`linked the demo in https://github.com/${REPO}/pull/${number}`);
  } finally {
    rmSync(script, { force: true });
    run("git", ["worktree", "remove", "--force", worktree]);
  }
}

async function recordDocs(): Promise<void> {
  const storage = getDemoStorage();

  if (!storage)
    throw new Error(
      "DEMO_STORAGE is not set: docs demo scripts are read from it",
    );

  const downloaded = join(runsDir(), "docs-source");

  rmSync(downloaded, { force: true, recursive: true });

  if (!storage.downloadDir("scripts/docs", downloaded)) {
    throw new Error("no scripts/docs/*.demo.ts in DEMO_STORAGE");
  }

  const scripts = readdirSync(downloaded)
    .filter((f) => f.endsWith(".demo.ts"))
    .map((f) =>
      writeDemoScript(
        basename(f, ".demo.ts"),
        readFileSync(join(downloaded, f), "utf8"),
      ),
    );

  try {
    const videos = await record(scripts);

    for (const { mp4 } of videos) {
      const name = basename(mp4, ".mp4");

      upload(storage, {
        [`videos/docs/${name}.gif`]: makePreview(mp4),
        [`videos/docs/${name}.mp4`]: mp4,
      });
    }
  } finally {
    rmSync(runsDir(), { force: true, recursive: true });
  }
}

/** Runs the command in process.argv for `demoApp`. */
export async function runDemoCli(demoApp: DemoApp): Promise<void> {
  app = demoApp;

  const envFile = join(app.dir, ".env");

  if (existsSync(envFile)) process.loadEnvFile(envFile);

  REPO = process.env.DEMO_GITHUB_REPO ?? "ggml-org/llama.cpp";

  const [command, ...args] = process.argv.slice(2);
  const trustIndex = args.indexOf("--trust");
  const extra = app.commands?.[command];

  if (command === "pr" && args[0]) {
    await recordPr(
      args[0],
      trustIndex === -1 ? undefined : args[trustIndex + 1],
    );
  } else if (command === "docs") {
    await recordDocs();
  } else if (extra) {
    await extra(args);
  } else {
    const others = Object.keys(app.commands ?? {}).map((name) => ` | ${name}`);

    console.error(
      `usage: node demo/cli.ts pr <number> [--trust <sha>] | docs${others.join("")}`,
    );
    process.exit(1);
  }
}
