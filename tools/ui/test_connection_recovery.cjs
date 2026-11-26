// A successful WHEP HTTP exchange must not leave ICE connecting forever.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const app = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
const code = app.slice(app.indexOf("class Player"), app.indexOf("function updateLive"));

function fixture() {
  let now = 0, sequence = 0;
  const timers = new Map(), nodes = new Map(), peers = [], requests = [];
  class Peer {
    constructor() { this.connectionState = "new"; peers.push(this); }
    addTransceiver() {}
    async createOffer() { return { type: "offer", sdp: "offer" }; }
    async setLocalDescription(value) { this.localDescription = value; }
    async setRemoteDescription() { this.connectionState = "connecting"; }
    close() { this.closed = true; this.connectionState = "closed"; }
  }
  const context = vm.createContext({
    config: { original_delay_ms: 0, h264_delay_ms: 0 }, paused: false,
    keyFor: (c, k) => `${c.id}-${k}`,
    $: (id) => {
      if (!nodes.has(id)) nodes.set(id, {
        addEventListener() {}, classList: { toggle() {} },
      });
      return nodes.get(id);
    },
    updateLive() {}, performance: { now: () => now },
    clearTimeout: (id) => timers.delete(id),
    setTimeout(fn, delay) { timers.set(++sequence, { fn, delay }); return sequence; },
    RTCPeerConnection: Peer, AbortController,
    gatherIce: async () => {},
    fetch: async (url, options) => {
      requests.push({ url, method: options.method });
      return {
        ok: true, headers: { get: () => `/api/session/${requests.length}` },
        text: async () => "answer",
      };
    },
  });
  vm.runInContext(code + "\nthis.Player = Player;", context);
  const create = (kind) => new context.Player({ id: "ch1" }, kind);
  const timer = (delay) => Array.from(timers.entries()).find(([, t]) => t.delay === delay);
  return {
    context, create, timer, timers, peers, requests,
    advance: (value) => { now = value; },
    run(id) { const t = timers.get(id); timers.delete(id); return t.fn(); },
  };
}

(async () => {
  for (const kind of ["original", "h264", "semantic"]) {
    const f = fixture(), player = f.create(kind);
    await player.connect();
    const first = player.peer;
    assert.equal(first.connectionState, "connecting");
    assert.ok(f.timer(15000), "HTTP completion must retain the first-frame deadline");
    await player.stats(); // No inbound stats exist while ICE is still connecting.
    const [deadline] = f.timer(15000);
    f.advance(16000);
    f.run(deadline);
    assert.equal(first.closed, true, "Timeout must release the failed peer");
    assert.equal(player.peer, null);
    assert.ok(f.requests.some(r => r.method === "DELETE"), "Release owned WHEP session");
    const [retry] = f.timer(3500);
    f.run(retry);
    await new Promise(setImmediate);
    assert.equal(f.peers.length, 2, "Retry must create a fresh connection");
    player.receivedFrame();
    assert.equal(f.timers.size, 0, "A first frame must cancel the deadline");
    player.close();
    assert.equal(f.timers.size, 0);
  }

  const stale = fixture(), player = stale.create("original");
  await player.connect();
  const callback = stale.timer(15000)[1].fn;
  await player.connect();
  const replacement = player.peer;
  callback();
  assert.equal(player.peer, replacement, "Old deadlines must not close replacements");
  assert.notEqual(replacement.closed, true);
  player.receivedFrame();
  assert.equal(stale.timers.size, 0);
  player.close();

  const multi = fixture(), original = multi.create("original"), semantic = multi.create("semantic");
  await original.connect();
  await semantic.connect();
  const healthy = semantic.peer;
  semantic.receivedFrame();
  multi.run(multi.timer(15000)[0]);
  assert.equal(semantic.peer, healthy, "One channel's timeout must not reconnect others");
  assert.notEqual(healthy.closed, true);
  original.close(); semantic.close();
  assert.equal(multi.timers.size, 0);
  console.log("Connection recovery: stalled ICE, first frame, stale deadlines and independent peers passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
