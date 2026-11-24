const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(__dirname + '/app.js', 'utf8');
const apiSource = source.slice(source.indexOf('async function api('), source.indexOf('function setState('));
const playerSource = source.slice(source.indexOf('class Player {'), source.indexOf('for(const c of channels)players'));
const reconnectSource = source.slice(source.indexOf('async function loadSettings('), source.indexOf('const format='));
const initializeSource = source.match(/async function initialize\(\).*$/m)[0];
const pipelineSource = source.slice(source.indexOf('async function pipeline('),source.indexOf("$('start-button').addEventListener"));
const config = {
  original:'rtsp://127.0.0.1:8554/camera-original',
  h264:'rtsp://127.0.0.1:8554/camera-h264',
  mlvc:'rtsp://127.0.0.1:8554/ulbvc', webrtc:'http://127.0.0.1:8889',
};

// Model the legacy RTSP pathname behavior shown in the customer's server log.
class LegacyURL extends URL {
  get pathname(){return this.protocol==='rtsp:'?'//'+this.host+super.pathname:super.pathname;}
}
function fixture(URLConstructor=URL) {
  const requests=[], notices=[], states={}, timers=new Map();
  let timerId=0;
  const sandbox={
    URL:URLConstructor, AbortController, config:{...config}, failure:null,
    channels:[{key:'original',name:'原图'},{key:'h264',name:'H.264'},{key:'mlvc',name:'语义压缩'}],
    players:{}, setState:(key,state)=>{states[key]=state;}, notice:message=>notices.push(message),
    fillSettings(){}, refresh:async()=>{}, iceReady:async()=>{}, $:()=>({disabled:false}),
    setTimeout:(callback,ms)=>{timers.set(++timerId,{callback,ms});return timerId;},
    clearTimeout:id=>timers.delete(id),
    RTCPeerConnection:class {
      constructor(){this.localDescription={sdp:'test offer'};}
      addTransceiver(){} async createOffer(){return {type:'offer',sdp:'test offer'};}
      async setLocalDescription(){} async setRemoteDescription(){}
    },
    fetch:async(url,options)=>{
      requests.push({url,options});
      if(sandbox.failure==='network')throw new Error('connection refused');
      if(sandbox.failure==='timeout')return new Promise((resolve,reject)=>{
        options.signal.addEventListener('abort',()=>{const error=new Error('aborted');error.name='AbortError';reject(error);},{once:true});
      });
      const invalidWhep=sandbox.failure==='whep'&&url.endsWith('/whep');
      return {ok:sandbox.failure!=='http'&&!invalidWhep,status:invalidWhep?400:503,
        json:async()=>{if(sandbox.failure==='json')throw new SyntaxError('invalid JSON');return sandbox.config;},
        headers:{get:()=>null},text:async()=>invalidWhep?'invalid path name':'test answer'};
    },
  };
  // No AbortSignal global is supplied: any use of AbortSignal.timeout fails.
  vm.createContext(sandbox);
  vm.runInContext('let settings={},configLoaded=false,enabled=true,statusSnapshot=null;'+apiSource+playerSource+reconnectSource+pipelineSource+initializeSource,sandbox);
  vm.runInContext(`for(const channel of channels){
    const player=Object.create(Player.prototype);player.key=channel.key;player.generation=0;
    player.close=function(){this.generation++;};players[channel.key]=player;
  }`,sandbox);
  return {sandbox,requests,notices,states,timers,run:code=>vm.runInContext(code,sandbox)};
}

(async()=>{
  const legacy=fixture();
  await legacy.run('initialize()');
  assert.equal(legacy.run('configLoaded'),true);
  assert.deepEqual(legacy.requests.filter(r=>r.url.endsWith('/whep')).map(r=>r.url),[
    'http://127.0.0.1:8889/camera-original/whep',
    'http://127.0.0.1:8889/camera-h264/whep',
    'http://127.0.0.1:8889/ulbvc/whep',
  ]);
  assert.equal(legacy.notices.length,0);
  assert.equal(legacy.timers.size,0,'completed requests must clear timers');

  assert.equal(new LegacyURL(config.original).pathname,'//127.0.0.1:8554/camera-original');
  const opaque=fixture(LegacyURL);
  await opaque.run('initialize()');
  assert.deepEqual(opaque.requests.filter(r=>r.url.endsWith('/whep')).map(r=>r.url),legacy.requests.filter(r=>r.url.endsWith('/whep')).map(r=>r.url));
  for(const [input,expected] of [
    ['RTSP://user:pass@192.168.5.3:8554/custom/semantic/?token=sample#fragment','custom/semantic'],
    ['rtsp://[::1]:8554/ulbvc','ulbvc'],
    ['rtsp://localhost:8554/camera-h264','camera-h264'],
    ['rtsp://localhost:8554/',''],
    ['http://localhost/ulbvc',''],
    [undefined,''],
  ])assert.equal(opaque.sandbox.pathFor(input),expected);

  const rejected=fixture();
  await rejected.run('loadSettings();');
  rejected.sandbox.failure='whep';
  rejected.run('statusSnapshot={feeds:{original:true,h264:true,mlvc:true}};');
  await rejected.run('reconnect()');
  assert.equal(rejected.notices.length,3);
  assert.ok(rejected.notices.every(message=>message.includes('WHEP 400：invalid path name')),'server rejection must remain visible in the UI');

  const retry=fixture();retry.sandbox.failure='network';
  await retry.run('initialize()');
  await retry.run('reconnect()');
  assert.equal(retry.run('configLoaded'),false);
  assert.deepEqual(Object.values(retry.states),['配置读取失败','配置读取失败','配置读取失败']);
  assert.ok(retry.notices.every(message=>message.includes('connection refused')));
  assert.equal(retry.requests.filter(r=>r.url.endsWith('/whep')).length,0);
  retry.sandbox.failure=null;
  await retry.run('reconnect()');
  assert.equal(retry.run('configLoaded'),true);
  assert.equal(retry.requests.filter(r=>r.url.endsWith('/whep')).length,3);

  const startRetry=fixture();startRetry.sandbox.failure='network';
  await startRetry.run('initialize()');startRetry.sandbox.failure=null;
  await startRetry.run("pipeline('start')");
  assert.equal(startRetry.requests.filter(r=>r.url.endsWith('/whep')).length,3,'starting H.264 after config recovery must connect every feed');

  const invalid=fixture();invalid.sandbox.config.mlvc='rtsp://127.0.0.1:8554';
  await invalid.run('initialize()');
  assert.equal(invalid.run('configLoaded'),false);
  assert.match(invalid.notices[0],/语义压缩 RTSP 配置无效/);
  assert.equal(invalid.requests.filter(r=>r.url.endsWith('/whep')).length,0);

  for(const failure of ['network','http','json']){
    const failed=fixture();failed.sandbox.failure=failure;
    await assert.rejects(failed.run("api('/api/config')"));
    assert.equal(failed.timers.size,0, failure+' must clear its timer');
  }
  const timeout=fixture();timeout.sandbox.failure='timeout';
  const request=timeout.run("api('/api/config')");
  const rejection=assert.rejects(request,error=>error.name==='AbortError');
  assert.equal(timeout.timers.size,1);
  const timer=[...timeout.timers.values()][0];assert.equal(timer.ms,6000);timer.callback();
  await rejection;assert.equal(timeout.timers.size,0);

  const post=fixture();
  await post.run("api('/api/config',{h264_qp:40})");
  assert.equal(post.requests[0].options.method,'POST');
  assert.equal(post.requests[0].options.body,'{"h264_qp":40}');
  console.log('Passed: no AbortSignal API, legacy RTSP pathname, three correct WHEP paths, credentials/IPv6/nested paths, config recovery, timeout/cleanup and POST.');
})().catch(error=>{console.error(error);process.exitCode=1;});
