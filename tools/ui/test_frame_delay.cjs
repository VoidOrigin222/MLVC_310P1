// Exercise buffer timing and frame lifetime without starting a browser.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const app = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
const code = app.slice(
  app.indexOf("class FrameDelay"),
  app.indexOf("function gatherIce"),
);

async function run(delay, fps, count, skew = 0, display = null) {
  const frames = [],
    drawings = [];
  let paint, clock = 0;
  const video = {
    readyState: 2,
    frame: 0,
    requestVideoFrameCallback(callback) {
      this.capture = callback;
      return 1;
    },
    cancelVideoFrameCallback() {
      this.capture = null;
    },
    after() {},
    clientWidth: display?.width,
    clientHeight: display?.height,
  };
  const canvas = {
    width: 0,
    height: 0,
    dataset: {},
    setAttribute() {},
    remove() {},
    getContext() {
      return {
        clearRect() {},
        drawImage(frame) {
          drawings.push({ frame: frame.id, at: video.frame });
        },
      };
    },
  };
  class FakeFrame {
    constructor(source) {
      this.id = source.frame;
      this.displayWidth = 1920;
      this.displayHeight = 1080;
      this.closes = 0;
      frames.push(this);
    }
    close() {
      this.closes++;
    }
  }
  const context = vm.createContext({
    document: { createElement: () => canvas },
    VideoFrame: FakeFrame,
    performance: { now: () => clock },
    requestAnimationFrame(callback) {
      paint = callback;
      return 1;
    },
    cancelAnimationFrame() {},
    Promise,
    devicePixelRatio: display?.pixelRatio || 1,
  });
  vm.runInContext(code + "\nthis.FrameDelay = FrameDelay;", context);
  const queue = new context.FrameDelay(video, delay);
  queue.start();
  for (let i = 0; i < count; i++) {
    video.frame = i;
    const now = (i * 1000) / fps;
    clock = now;
    video.capture(now - skew, { expectedDisplayTime: now + skew });
    await new Promise(setImmediate);
    paint(now - skew);
  }
  assert.ok(drawings.length > 0, `Black picture: ${delay} ms at ${fps} FPS`);
  assert.equal(canvas.width, 1920, "Keep full source resolution even in small cards");
  assert.equal(canvas.height, 1080);
  assert.ok(Number(canvas.dataset.delayMs) >= 0, "Negative display delay");
  if (fps <= 60) {
    const last = drawings.at(-1);
    assert.ok(((last.at - last.frame) * 1000) / fps >= delay - 0.001);
  }
  if (display) {
    video.clientWidth = 1920;
    video.clientHeight = 1080;
    video.frame++;
    clock += 1000;
    paint(clock);
    assert.equal(canvas.width, 1920, "Fullscreen must restore source resolution");
    assert.equal(canvas.height, 1080);
  }
  queue.stop();
  assert.ok(
    frames.every((frame) => frame.closes === 1),
    "Frame leak or double close",
  );
}

(async () => {
  await run(200, 13, 60);
  await run(2000, 30, 130);
  await run(2000, 240, 600);
  await run(200, 12, 100, 10000);
  await run(200, 30, 100, 0, { width: 320, height: 180 });
  await run(200, 30, 100, 0, { width: 320, height: 180, pixelRatio: 2 });
  console.log(
    "Frame delay: timing, long compensation and bounded lifetime passed",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
