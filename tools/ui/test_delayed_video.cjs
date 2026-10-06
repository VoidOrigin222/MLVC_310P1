const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(__dirname + '/app.js', 'utf8');
const rendererSource = source.slice(source.indexOf('class DelayedVideo {'), source.indexOf('class Player {'));

function fixture(useBitmap = false) {
  let now = 0, id = 0, resolveBitmap;
  const callbacks = new Map(), frames = [], draws = [];
  class Frame {
    constructor() { this.displayWidth = 1920; this.displayHeight = 1080; this.closed = false; frames.push(this); }
    close() { assert.equal(this.closed, false, 'frame released twice'); this.closed = true; }
  }
  const context = {drawImage: frame => draws.push(frame), clearRect() {}};
  const canvas = {width: 300, height: 150, setAttribute() {}, getContext: () => context};
  const video = {
    dataset: {}, readyState: 4, videoWidth: 1920,
    insertAdjacentElement() {},
    requestVideoFrameCallback: callback => { callbacks.set(++id, callback); return id; },
    cancelVideoFrameCallback: request => callbacks.delete(request),
  };
  const sandbox = {
    document: {createElement: () => canvas},
    performance: {now: () => now},
    requestAnimationFrame: callback => { callbacks.set(++id, callback); return id; },
    cancelAnimationFrame: request => callbacks.delete(request),
    notice: () => assert.fail('unexpected rendering failure'),
    VideoFrame: useBitmap ? undefined : Frame,
    createImageBitmap: () => new Promise(resolve => { resolveBitmap = resolve; }),
  };
  const Renderer = vm.runInNewContext(rendererSource + '\nDelayedVideo;', sandbox);
  const renderer = new Renderer(video, 200);
  renderer.start();
  return {renderer, video, frames, draws, callbacks, capture(time, expectedDisplayTime) {
    now = time;
    const callback = callbacks.get(renderer.frameRequest);
    callbacks.delete(renderer.frameRequest);
    callback(time, {expectedDisplayTime});
  }, resolveBitmap: () => resolveBitmap(new Frame())};
}

// A decoded frame must stay invisible until its full 200 ms hold has elapsed.
{
  const f = fixture();
  f.capture(0); f.renderer.paint(199);
  assert.equal(f.draws.length, 0);
  f.renderer.paint(200);
  assert.equal(f.draws.length, 1);
  assert.equal(JSON.parse(f.video.dataset.presentationDelayStats).actualMs, 200);
  f.renderer.stop();
  assert.equal(f.callbacks.size, 0);
}

// Preserve the native frame presentation baseline, then add the full requested delay.
{
  const f = fixture();
  f.capture(0, 16); f.renderer.paint(215);
  assert.equal(f.draws.length, 0);
  f.renderer.paint(216);
  assert.equal(f.draws.length, 1);
  assert.equal(JSON.parse(f.video.dataset.presentationDelayStats).actualMs, 200);
  f.renderer.stop();
}

// Sustained video retains its cadence and bounded delay rather than accumulating latency.
{
  const f = fixture();
  for (let i = 0; i < 90; i++) { f.capture(i * 34); f.renderer.paint(i * 34); }
  const stats = JSON.parse(f.video.dataset.presentationDelayStats);
  assert.equal(f.draws.length, 84);
  assert.equal(stats.actualMs, 204);
  assert.equal(stats.skippedFrames, 0);
  assert.equal(stats.queuedFrames, 6);
  f.renderer.stop();
  assert.ok(f.frames.every(frame => frame.closed));
}

// A stalled paint loop cannot grow memory forever; resume selects the latest eligible frame.
{
  const f = fixture();
  for (let i = 0; i < 60; i++) f.capture(i * 34);
  assert.equal(f.renderer.queue.length, 12);
  f.renderer.paint(60 * 34);
  assert.ok(JSON.parse(f.video.dataset.presentationDelayStats).actualMs < 234);
  f.renderer.stop();
  assert.ok(f.frames.every(frame => frame.closed));
}

// An asynchronous bitmap finishing after disconnect must be released, never replayed.
(async () => {
  const f = fixture(true);
  f.capture(0); f.renderer.stop(); f.resolveBitmap();
  await Promise.resolve(); await Promise.resolve();
  assert.equal(f.renderer.queue.length, 0);
  assert.ok(f.frames.every(frame => frame.closed));
  assert.equal(f.draws.length, 0);
  console.log('Passed: 200 ms hold, steady cadence, bounded backlog, disconnect cleanup.');
})().catch(error => { console.error(error); process.exitCode = 1; });
