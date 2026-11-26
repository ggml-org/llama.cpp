/**
 * Drives a page like a person for a demo video: a visible cursor or fingertip, human timing,
 * captions, a voice, and camera zooms that the render plays (see polish.ts). Playwright is only a
 * type here: the app passes in its page, so this package loads none of its own.
 */

import { CAMERA_MOVE_MS, type PolishEvent } from "./polish.ts";
import type { CDPSession, Locator, Page } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/** How far below the viewport's top a pinch has to start for Chromium to zoom at all. */
const ZOOM_TOP_INSET_PX = 64;
/** The timeline of each page's demo, which the polished render plays over the footage. */
export const timelines = new WeakMap<Page, PolishEvent[]>();

/**
 * Draws a fake cursor and click ripples, since recorded videos don't include
 * the OS pointer, and a fingertip per touch, so a pinch shows as two fingers.
 * A touch device has no cursor: a click shows as a tap of a fingertip.
 * Runs in the page via addInitScript.
 */
export function installCursorOverlay(): void {
  const mount = () => {
    const style = document.createElement("style");

    style.textContent = `
			#demo-cursor { position: fixed; top: 0; left: 0; z-index: 2147483647; pointer-events: none;
				width: 22px; height: 22px; margin: -3px 0 0 -3px; transition: transform 80ms ease-out; }
			#demo-cursor.down { transform: scale(0.85); }
			.demo-ripple { position: fixed; z-index: 2147483646; pointer-events: none; width: 36px; height: 36px;
				margin: -18px 0 0 -18px; border-radius: 50%; background: rgba(59, 130, 246, 0.35);
				animation: demo-ripple 450ms ease-out forwards; }
			@keyframes demo-ripple { from { transform: scale(0.3); opacity: 1; } to { transform: scale(1.4); opacity: 0; } }
			#demo-caption { position: fixed; left: 50%; top: 72px; z-index: 2147483647; pointer-events: none;
				transform: translate(-50%, -8px); opacity: 0; transition: opacity 200ms ease, transform 200ms ease;
				max-width: 80%; padding: 10px 18px; border-radius: 10px; background: rgba(17, 17, 17, 0.88);
				color: #fff; font: 500 18px/1.4 system-ui, sans-serif; text-align: center; }
			#demo-caption.visible { opacity: 1; transform: translate(-50%, 0); }
			.demo-finger { position: fixed; z-index: 2147483646; pointer-events: none; width: 40px; height: 40px;
				margin: -20px 0 0 -20px; border-radius: 50%; background: rgba(59, 130, 246, 0.3);
				border: 2px solid rgba(59, 130, 246, 0.55); transition: opacity 200ms ease; }
		`;
    document.head.appendChild(style);

    const cursor = document.createElement("div");

    cursor.id = "demo-cursor";
    cursor.innerHTML = `<svg viewBox="0 0 24 24" width="22" height="22"><path d="M4 2l16 10.5-7 1.5-3.5 6.5z"
			fill="#111" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/></svg>`;
    document.documentElement.appendChild(cursor);

    const caption = document.createElement("div");

    caption.id = "demo-caption";
    document.documentElement.appendChild(caption);

    const touch = navigator.maxTouchPoints > 0;
    const tap = Object.assign(document.createElement("div"), {
      className: "demo-finger",
    });
    let lift: ReturnType<typeof setTimeout> | undefined;

    if (touch) cursor.style.display = "none";

    document.addEventListener(
      "mousemove",
      (e) => (cursor.style.translate = `${e.clientX}px ${e.clientY}px`),
      true,
    );
    document.addEventListener(
      "mousedown",
      (e) => {
        cursor.classList.add("down");

        if (touch) {
          tap.style.left = `${e.clientX}px`;
          tap.style.top = `${e.clientY}px`;
          tap.style.opacity = "1";
          document.documentElement.appendChild(tap);
          // a quick second tap restarts the fade, so the first one's timer does not hide it
          clearTimeout(lift);
          lift = setTimeout(() => (tap.style.opacity = "0"), 250);
        }

        const ripple = document.createElement("div");

        ripple.className = "demo-ripple";
        ripple.style.left = `${e.clientX}px`;
        ripple.style.top = `${e.clientY}px`;
        document.documentElement.appendChild(ripple);
        setTimeout(() => ripple.remove(), 500);
      },
      true,
    );
    document.addEventListener(
      "mouseup",
      () => cursor.classList.remove("down"),
      true,
    );

    const fingers: HTMLDivElement[] = [];
    const placeFingers = (e: TouchEvent) => {
      Array.from(e.touches).forEach((touch, i) => {
        const finger = (fingers[i] ??= Object.assign(
          document.createElement("div"),
          {
            className: "demo-finger",
          },
        ));

        finger.style.left = `${touch.clientX}px`;
        finger.style.top = `${touch.clientY}px`;
        finger.style.opacity = "1";
        document.documentElement.appendChild(finger);
      });
      fingers
        .slice(e.touches.length)
        .forEach((finger) => (finger.style.opacity = "0"));
    };

    for (const type of ["touchstart", "touchmove", "touchend", "touchcancel"]) {
      document.addEventListener(type, placeFingers as EventListener, {
        capture: true,
        passive: true,
      });
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
}

/**
 * Swaps the page's microphone for one that hears only what `window.demoSpeak` plays into it,
 * silence otherwise. Chromium's fake capture device crashes the page, so voice input is
 * recorded with this instead. Runs in the page.
 */
function installFakeMicrophone(): void {
  const context = new AudioContext();
  const output = context.createMediaStreamDestination();

  navigator.mediaDevices.getUserMedia = async (constraints) => {
    if (!constraints?.audio || constraints.video) {
      throw new DOMException(
        "demo: only the microphone is faked",
        "NotSupportedError",
      );
    }

    await context.resume();

    // each opening gets its own tracks, so a recording that stops them leaves the source on
    return new MediaStream(
      output.stream.getAudioTracks().map((track) => track.clone()),
    );
  };

  Object.assign(window, {
    demoSpeak: async (wavBase64: string) => {
      const bytes = Uint8Array.from(atob(wavBase64), (c) => c.charCodeAt(0));
      const source = context.createBufferSource();

      source.buffer = await context.decodeAudioData(bytes.buffer);
      source.connect(output);
      await context.resume();
      await new Promise((done) => {
        source.onended = done;
        source.start();
      });
    },
  });
}

/** `text` read by a synthetic voice, as a base64 WAV. Uses macOS `say`. */
function synthesizeSpeech(text: string, voice: string): string {
  const dir = mkdtempSync(join(tmpdir(), "demo-speech-"));
  const file = join(dir, "speech.wav");

  try {
    execFileSync("say", [
      "-v",
      voice,
      "-o",
      file,
      "--file-format=WAVE",
      "--data-format=LEI16@24000",
      text,
    ]);

    return readFileSync(file).toString("base64");
  } catch (error) {
    throw new Error(
      `demo: speak() needs macOS \`say\` with the voice ${voice}`,
      {
        cause: error,
      },
    );
  } finally {
    rmSync(dir, { force: true, recursive: true });
  }
}

/** Seeded randomness, so a demo moves the same way on every recording. */
function createRandom(seed: string): () => number {
  let state = [...seed].reduce(
    (h, c) => Math.imul(h ^ c.charCodeAt(0), 2654435761),
    1,
  );

  return () => {
    state = (state + 0x6d2b79f5) | 0;

    let t = Math.imul(state ^ (state >>> 15), 1 | state);

    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const easeInOut = (t: number) =>
  t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;

/**
 * Drives the page like a person: the cursor glides along a slight arc with
 * Fitts-like durations, settles briefly before pressing, and typing and pauses
 * vary a little.
 */
export class Demo {
  private cdp: CDPSession | undefined;
  private readonly page: Page;
  private readonly random: () => number;
  private readonly timeline: PolishEvent[] | undefined;
  private x = 0;
  private y = 0;

  /** Shows a caption at the top of the frame for `ms`, clear of the chat form. */
  async caption(text: string, ms = 2200): Promise<void> {
    await this.showCaption(text);
    await this.page.waitForTimeout(ms);
    await this.hideCaption();
  }

  async click(target: Locator, ms = 400): Promise<void> {
    await this.moveTo(target);
    await this.page.waitForTimeout(this.between(80, 160));
    await this.page.mouse.down();
    await this.page.waitForTimeout(this.between(50, 90));
    await this.page.mouse.up();
    await this.pause(ms);
  }

  constructor(page: Page, seed: string) {
    this.page = page;
    this.random = createRandom(seed);
    this.timeline = timelines.get(page);

    const viewport = page.viewportSize();

    this.x = (viewport?.width ?? 0) / 2;
    this.y = (viewport?.height ?? 0) / 2;
  }

  /**
   * Touches `target` for `ms` and lifts: a tap, or with a longer `ms` a press-and-hold, as a
   * phone's held rows ask for. It needs a touch device (`hasTouch`), and shows a fingertip.
   */
  async hold(target: Locator, ms = 900, after = 500): Promise<void> {
    await target.scrollIntoViewIfNeeded();

    const { x, y } = await this.center(target);
    const cdp = await this.session();

    await this.page.waitForTimeout(this.between(60, 140));
    await cdp.send("Input.dispatchTouchEvent", {
      touchPoints: [{ x, y }],
      type: "touchStart",
    });
    await this.page.waitForTimeout(ms);
    await cdp.send("Input.dispatchTouchEvent", {
      touchPoints: [],
      type: "touchEnd",
    });
    await this.liftFingers();
    this.x = x;
    this.y = y;
    await this.pause(after);
  }

  async hover(target: Locator, ms = 350): Promise<void> {
    await this.moveTo(target);
    await this.pause(ms);
  }

  /**
   * Gives the page a microphone that hears only what `speak` says, and allows the page to
   * use it. Call it before the microphone is first used.
   */
  async microphone(): Promise<void> {
    await this.page.context().grantPermissions(["microphone"]);
    await this.page.evaluate(installFakeMicrophone);
  }

  /** Glides the cursor to a point near the center of `target`. */
  async moveTo(target: Locator): Promise<void> {
    await target.scrollIntoViewIfNeeded();
    const box = await target.boundingBox();

    if (!box) throw new Error(`demo: ${target} has no bounding box`);

    const x = box.x + box.width * this.between(0.4, 0.6);
    const y = box.y + box.height * this.between(0.4, 0.6);
    const dx = x - this.x;
    const dy = y - this.y;
    const distance = Math.hypot(dx, dy);
    const duration = 110 + 55 * Math.log2(distance / 40 + 1);
    // bow the path a little to one side, like a wrist turning
    const bow = distance * this.between(-0.08, 0.08);
    const fromX = this.x;
    const fromY = this.y;
    const started = Date.now();

    for (;;) {
      const t = Math.min((Date.now() - started) / duration, 1);
      const p = easeInOut(t);
      const arc = Math.sin(Math.PI * p) * bow;

      this.x = fromX + dx * p - (dy / (distance || 1)) * arc;
      this.y = fromY + dy * p + (dx / (distance || 1)) * arc;
      await this.page.mouse.move(this.x, this.y);

      if (t === 1) break;

      await this.page.waitForTimeout(8);
    }
  }

  /** Waits about `ms`, varied by up to 15% so pauses don't feel metronomic. */
  async pause(ms = 800): Promise<void> {
    await this.page.waitForTimeout(ms * this.between(0.85, 1.15));
  }

  async press(key: string, ms = 600): Promise<void> {
    await this.page.keyboard.press(key);
    await this.pause(ms);
  }

  /** Scrolls the wheel over `target` by `dy` pixels (negative scrolls up), in small steps like a trackpad. */
  async scroll(target: Locator, dy: number, ms = 500): Promise<void> {
    await this.moveTo(target);

    const step = Math.sign(dy) * 40;

    for (let done = 0; Math.abs(done) < Math.abs(dy); done += step) {
      await this.page.mouse.wheel(0, step);
      await this.page.waitForTimeout(16);
    }

    await this.pause(ms);
  }

  /**
   * Says `text` into the microphone from `microphone` in a synthetic voice (macOS `say`), and
   * returns once it is said. The words show as a caption, since the video has no sound.
   */
  async speak(text: string, voice = "Samantha", ms = 400): Promise<void> {
    const wav = synthesizeSpeech(text, voice);

    await this.showCaption(`"${text}"`);
    await this.page.evaluate(
      (base64) =>
        (
          window as unknown as { demoSpeak(wav: string): Promise<void> }
        ).demoSpeak(base64),
      wav,
    );
    await this.hideCaption();
    await this.pause(ms);
  }

  /** Taps `target` with a fingertip, on a touch device. */
  tap(target: Locator, after = 500): Promise<void> {
    return this.hold(target, this.between(60, 110), after);
  }

  /** Types quickly with uneven key timing, a little slower after spaces and punctuation. */
  async type(target: Locator, text: string, ms = 500): Promise<void> {
    await this.click(target, 200);

    for (const char of text) {
      await this.page.keyboard.type(char);
      await this.page.waitForTimeout(
        this.between(15, 45) +
          (/[\s.,!?]/.test(char) ? this.between(20, 60) : 0),
      );
    }

    await this.pause(ms);
  }

  /**
   * Pinches in on `target`, a control or a point in the viewport, holds about `ms`, and pinches
   * back out to 1x: a two-finger zoom onto a detail too small to read in the frame. On a touch
   * device the fingertips show. In the polished render the camera moves in and out instead.
   */
  async zoom(
    target: Locator | { x: number; y: number },
    scale = 2.5,
    ms = 1400,
  ): Promise<void> {
    if (this.timeline) {
      await this.zoomIn(target, scale);
      await this.pause(ms);
      await this.zoomOut();

      return;
    }

    const point = "x" in target ? target : await this.center(target);
    // a pinch anchored in the viewport's top band does not zoom at all
    const at = { x: point.x, y: Math.max(point.y, ZOOM_TOP_INSET_PX) };
    const viewport = this.page.viewportSize() ?? { height: 0, width: 0 };
    const middle = { x: viewport.width / 2, y: viewport.height / 2 };

    await this.pinch(at, scale, 1400);

    // One near the bottom edge overshoots its ratio, and a pinch there again misses too. A small
    // pinch from the middle evens it out, drawing the detail toward the middle, so it stays in view
    const reached = await this.pageScale();

    if (Math.abs(reached / scale - 1) > 0.03)
      await this.pinch(middle, scale / reached, 1400);

    await this.liftFingers();
    await this.pause(ms);

    // back to exactly 1x, which the next gesture needs, in one quick pinch from the middle of
    // the viewport, and another for what is left when one lands short
    for (let tries = 0; ; tries++) {
      const current = await this.pageScale();

      if (current <= 1.001) break;

      if (tries === 4)
        throw new Error(`demo: zoom stuck at ${current.toFixed(3)}x`);

      await this.pinch(middle, 1 / current, 1800);
    }

    await this.liftFingers();
    await this.pause(200);
  }

  /**
   * Moves the camera in on `target` and keeps it there while the demo goes on,
   * until `zoomOut` or the next `zoomIn`. A plain recording (DEMO_POLISH=0) ignores it, as the cursor cannot act
   * on a pinched page.
   */
  async zoomIn(
    target: Locator | { x: number; y: number },
    scale = 2.5,
  ): Promise<void> {
    if (!this.timeline) return;

    if (!("x" in target)) await target.scrollIntoViewIfNeeded();

    const point = "x" in target ? target : await this.center(target);

    this.timeline.push({
      scale,
      time: Date.now() / 1000,
      type: "camera",
      ...point,
    });
    await this.page.waitForTimeout(CAMERA_MOVE_MS);
  }

  /** Moves the camera back out to the whole page. */
  async zoomOut(): Promise<void> {
    if (!this.timeline) return;

    this.timeline.push({
      scale: 1,
      time: Date.now() / 1000,
      type: "camera",
      x: 0,
      y: 0,
    });
    await this.page.waitForTimeout(CAMERA_MOVE_MS);
  }

  private between(min: number, max: number): number {
    return min + (max - min) * this.random();
  }

  private async center(target: Locator): Promise<{ x: number; y: number }> {
    const box = await target.boundingBox();

    if (!box) throw new Error(`demo: ${target} has no bounding box`);

    return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  }

  private async hideCaption(): Promise<void> {
    this.timeline?.push({
      text: null,
      time: Date.now() / 1000,
      type: "caption",
    });
    await this.page.evaluate(() =>
      document.getElementById("demo-caption")?.classList.remove("visible"),
    );
    await this.page.waitForTimeout(250);
  }

  /** A pinch that reaches the screen's edge can end without its last touchend reaching the page. */
  private async liftFingers(): Promise<void> {
    await this.page.evaluate(() =>
      document
        .querySelectorAll<HTMLElement>(".demo-finger")
        .forEach((finger) => (finger.style.opacity = "0")),
    );
  }

  private pageScale(): Promise<number> {
    return this.page.evaluate(() => window.visualViewport?.scale ?? 1);
  }

  /**
   * A two-finger pinch by `factor` around `at`, in viewport pixels. While zoomed in, the gesture
   * reads its position in page-scale units, so it is divided by the scale the page is at.
   */
  private async pinch(
    at: { x: number; y: number },
    factor: number,
    speed: number,
  ): Promise<void> {
    const cdp = await this.session();
    const current = await this.pageScale();

    await cdp.send("Input.synthesizePinchGesture", {
      gestureSourceType: "touch",
      relativeSpeed: speed,
      scaleFactor: factor,
      x: at.x / current,
      y: at.y / current,
    });
  }

  private async session(): Promise<CDPSession> {
    this.cdp ??= await this.page.context().newCDPSession(this.page);

    return this.cdp;
  }

  private async showCaption(text: string): Promise<void> {
    if (this.timeline) {
      this.timeline.push({ text, time: Date.now() / 1000, type: "caption" });

      return;
    }

    await this.page.evaluate((t) => {
      const el = document.getElementById("demo-caption");

      if (!el) return;

      el.textContent = t;
      el.classList.add("visible");
    }, text);
  }
}
