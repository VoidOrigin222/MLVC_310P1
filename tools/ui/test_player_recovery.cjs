// Exercise reconnection against absent reports, delayed keyframes and recovery.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const app = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
const code = app.slice(app.indexOf("class Player"), app.indexOf("function updateLive"));

function fixture() {
  let now = 0, timerId = 0;
  const timers = new Map(), nodes = new Map();
  const video = {
    addEventListener() {},
    videoWidth: 1920,
    videoHeight: 1080,
    srcObject: null,
  };
  const context = vm.createContext({
    config: { original_delay_ms: 0, h264_delay_ms: 0 },
    paused: false,
    keyFor: (c, k) => `${c.id}-${k}`,
    $: (id) => {
      if (id.endsWith("-video")) return video;
      if (!nodes.has(id)) nodes.set(id, { classList: { toggle() {} } });
      return nodes.get(id);
    },
    updateLive() {},
    number: (v) => String(v),
    performance: { now: () => now },
    clearTimeout: (id) => timers.delete(id),
    setTimeout: (fn) => { timers.set(++timerId, fn); return timerId; },
  });
  vm.runInContext(code + "\nthis.Player = Player;", context);
  const player = new context.Player({ id: "ch1" }, "semantic");
  player.close();
  let reports = [];
  player.peer = { connectionState: "connected", getStats: async () => new Map(reports.map((r, i) => [i, r])) };
  return {
    player, timers,
    time: (value) => { now = value; },
    reports: (value) => { reports = value; },
  };
}

const report = (time, frames) => ({
  type: "inbound-rtp", kind: "video", timestamp: time,
  framesDecoded: frames, framesPerSecond: 12, bytesReceived: frames * 1000,
});

(async () => {
  // A connected ICE transport alone does not guarantee a video track/report.
  const absent = fixture();
  absent.time(16000);
  await absent.player.stats();
  assert.equal(absent.timers.size, 1, "No-report session must reconnect");

  // At 12 FPS a 96-frame GOP can take eight seconds to supply a keyframe.
  const first = fixture();
  first.time(8000);
  first.reports([report(8000, 0)]);
  await first.player.stats();
  assert.equal(first.timers.size, 0, "Do not interrupt first-keyframe wait");
  first.time(16000);
  first.reports([report(16000, 0)]);
  await first.player.stats();
  assert.equal(first.timers.size, 1, "First-frame wait must remain bounded");

  // Fresh frames must cancel the pending reconnect and remove the empty overlay.
  first.time(17000);
  first.reports([report(17000, 12)]);
  await first.player.stats();
  assert.equal(first.timers.size, 0);
  assert.equal(first.player.live, true);
  first.time(24000);
  first.reports([report(24000, 12)]);
  await first.player.stats();
  assert.equal(first.timers.size, 1, "Stalled established stream must reconnect");
  first.time(25000);
  first.reports([report(25000, 24)]);
  await first.player.stats();
  assert.equal(first.timers.size, 0, "Recovered stream must keep its session");
  assert.equal(first.player.live, true);
  console.log("Player recovery: absent reports, first frame and resumed frames passed");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
