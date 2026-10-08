const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(__dirname + '/app.js', 'utf8');
const start = source.indexOf('function fillSettings()');
const end = source.indexOf('function clock()', start);
const fields = Object.fromEntries(['original','h264','mlvc','webrtc'].map(name => [name, {
  name, type:'text', value:'', matches:()=>true,
}]));
const listeners = {};
const elements = {...fields,
  'settings-dialog': {addEventListener(){}, close(){}},
  'settings-form': {addEventListener(event, callback){listeners[event]=callback;}, querySelectorAll:()=>Object.values(fields)},
  'settings-error': {}, 'save-settings': {},
  'cancel-settings': {addEventListener(){}},
  'close-settings': {addEventListener(){}},
};
const calls = [];
const sandbox = {
  URL, Object, Number, String,
  settings: {original:'rtsp://127.0.0.1:8554/camera-original',h264:'rtsp://127.0.0.1:8554/camera-h264',
    mlvc:'rtsp://127.0.0.1:8554/ulbvc',webrtc:'http://127.0.0.1:8889',mode:'same_quality',h264_qp:40},
  enabled:false, $:id=>elements[id],
  document:{querySelectorAll:()=>[]},
  api:async (path,body)=>calls.push({path,body}), reconnect(){}, notice(){},
};
vm.createContext(sandbox);
vm.runInContext(source.slice(start,end),sandbox);
(async()=>{
  sandbox.fillSettings();
  assert.equal(fields.mlvc.value,'rtsp://127.0.0.1:8554/ulbvc','full stream path must remain visible');
  fields.mlvc.value='rtsp://192.168.10.20:8554/custom/semantic?token=sample';
  await listeners.submit({preventDefault(){}});
  assert.equal(calls.length,1);
  assert.equal(calls[0].path,'/api/config');
  assert.equal(calls[0].body.mlvc,fields.mlvc.value,'saving must preserve the new stream path and query');
  sandbox.fillSettings();
  assert.equal(fields.mlvc.value,'rtsp://192.168.10.20:8554/custom/semantic?token=sample');
  const pathSource=source.slice(source.indexOf('function pathFor('),source.indexOf('function setState('));
  vm.runInContext(pathSource,sandbox);
  assert.equal(sandbox.pathFor(sandbox.settings.mlvc),'custom/semantic','WHEP must use the configured path');
  console.log('Passed: complete RTSP display, save round trip and configured WHEP path.');
})().catch(error=>{console.error(error);process.exitCode=1;});
