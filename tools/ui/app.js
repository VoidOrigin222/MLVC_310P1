'use strict';
const $ = id => document.getElementById(id);
const channels = [
  {key:'original', name:'原图', tag:'原始输入', path:'camera-original', hint:'等待摄像头原始流'},
  {key:'h264', name:'H.264', tag:'编码对照', path:'camera-h264', hint:'等待 H.264 推流'},
  {key:'mlvc', name:'语义压缩', tag:'解码输出', path:'mlvc', hint:'等待解码板视频流'},
];
const cameraIcon = '<svg viewBox="0 0 48 48"><rect x="6" y="13" width="27" height="23" rx="4"/><path d="m33 20 10-5v20l-10-5M17 13l3-5h10l3 5"/><circle cx="19.5" cy="24.5" r="5.5"/></svg>';
const expandIcon = '<svg viewBox="0 0 24 24"><path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/></svg>';
for (const [index,c] of channels.entries()) {
  $('video-grid').insertAdjacentHTML('beforeend', `<article class="video-card ${c.key}" id="${c.key}-card"><div class="video-header"><h3 class="video-name">${c.name}</h3><span class="video-state"><i class="status-dot"></i><span id="${c.key}-state">未连接</span></span><span class="video-resolution" id="${c.key}-res">— × — · — FPS</span></div><div class="video-wrap" id="${c.key}-wrap"><video id="${c.key}-video" autoplay muted playsinline aria-label="${c.name}实时视频"></video><div class="video-empty">${cameraIcon}<strong id="${c.key}-empty">等待视频信号</strong></div><button class="fullscreen" data-fullscreen="${c.key}" aria-label="${c.name}全屏" title="全屏">${expandIcon}</button></div></article>`);
  $(c.key+'-card').insertAdjacentHTML('beforeend', `<div class="metric-card"><div class="metric"><span class="metric-label">bpp</span><div class="metric-value"><b id="${c.key}-bpp">—</b></div></div><div class="metric"><span class="metric-label">带宽</span><div class="metric-value"><b id="${c.key}-kbs">—</b><span>kB/s</span></div></div><div class="metric"><span class="metric-label">压缩比</span><div class="metric-value"><b id="${c.key}-ratio">—</b><span>: 1</span></div></div></div>`);
}
let settings = {}, enabled = true, refreshing = false, statusSnapshot = null;
const players = {};
function notice(message, error=false) {$('notice').hidden=false;$('notice').textContent=String(message).replace(/mlvc/gi,'语义压缩');$('notice').className='notice'+(error?' error':'');}
async function api(path, body) {
  const response=await fetch(path,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),signal:AbortSignal.timeout(6000)});
  const result=await response.json();if(!response.ok)throw new Error(result.error||'请求失败：'+response.status);return result;
}
function pathFor(url) {try{return new URL(url).pathname.replace(/^\/+|\/+$/g,'');}catch{return '';}}
function setState(key, text, live=false) {
  const card=$(key+'-card');
  $(key+'-state').textContent=text;card.classList.toggle('live',live);
  $(key+'-empty').textContent=text==='未发布'?'等待视频信号':text;
  $('live-count').textContent=document.querySelectorAll('.video-card.live').length+' / 3 在线';
}
function iceReady(pc, signal) {
  if(pc.iceGatheringState==='complete')return Promise.resolve();
  return new Promise(resolve=>{const finish=()=>{clearTimeout(timer);pc.removeEventListener('icegatheringstatechange',check);signal.removeEventListener('abort',finish);resolve();};const check=()=>{if(pc.iceGatheringState==='complete')finish();};const timer=setTimeout(finish,1500);pc.addEventListener('icegatheringstatechange',check);signal.addEventListener('abort',finish,{once:true});});
}
// Hold original and H.264 frames for 200 ms; semantic video plays directly.
class DelayedVideo {
  constructor(video, delayMs) {
    this.video=video;this.delayMs=delayMs;this.queue=[];this.active=false;this.epoch=0;
    this.frameRequest=null;this.captureRequest=null;this.paintRequest=null;this.inFlight=0;
    this.canvas=document.createElement('canvas');this.canvas.className='delayed-video';
    this.canvas.setAttribute('aria-hidden','true');video.insertAdjacentElement('afterend',this.canvas);
    this.context=this.canvas.getContext('2d',{alpha:false,desynchronized:true});
    this.presented=0;this.skipped=0;this.lastMediaTime=null;
  }
  start() {
    if(this.active)return;
    this.active=true;const epoch=++this.epoch;this.presented=0;this.skipped=0;
    const capture=(now,metadata)=>{
      if(!this.active||epoch!==this.epoch)return;
      this.scheduleCapture(capture);
      if(this.video.readyState<2||!this.video.videoWidth||this.inFlight>=2)return;
      if(typeof this.video.requestVideoFrameCallback!=='function'){
        if(this.lastMediaTime===this.video.currentTime)return;
        this.lastMediaTime=this.video.currentTime;
      }
      const capturedAt=Number.isFinite(metadata?.expectedDisplayTime)?metadata.expectedDisplayTime:performance.now();this.inFlight++;
      try {
        const frame=typeof VideoFrame==='function'?new VideoFrame(this.video,{timestamp:Math.round(capturedAt*1000)}):createImageBitmap(this.video);
        if(frame&&typeof frame.then==='function'){
          frame.then(value=>this.enqueue(value,capturedAt,epoch),error=>this.fail(error,epoch)).finally(()=>{if(epoch===this.epoch)this.inFlight--;});
        } else {this.inFlight--;this.enqueue(frame,capturedAt,epoch);}
      } catch(error){this.inFlight--;this.fail(error,epoch);}
    };
    const paint=now=>{
      if(!this.active||epoch!==this.epoch)return;
      this.paint(now);
      this.paintRequest=requestAnimationFrame(paint);
    };
    this.scheduleCapture(capture);this.paintRequest=requestAnimationFrame(paint);
  }
  scheduleCapture(callback) {
    if(typeof this.video.requestVideoFrameCallback==='function')this.frameRequest=this.video.requestVideoFrameCallback(callback);
    else this.captureRequest=requestAnimationFrame(callback);
  }
  enqueue(frame,capturedAt,epoch) {
    if(!this.active||epoch!==this.epoch){frame.close();return;}
    this.queue.push({frame,capturedAt});this.queue.sort((a,b)=>a.capturedAt-b.capturedAt);
    while(this.queue.length>12){this.queue.shift().frame.close();this.skipped++;}
  }
  paint(now) {
    let selected=null;
    while(this.queue.length&&now-this.queue[0].capturedAt>=this.delayMs){
      if(selected){selected.frame.close();this.skipped++;}
      selected=this.queue.shift();
    }
    if(!selected)return;
    try {
      const width=selected.frame.displayWidth||selected.frame.width,height=selected.frame.displayHeight||selected.frame.height;
      if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
      this.context.drawImage(selected.frame,0,0,width,height);this.presented++;
      this.video.dataset.presentationDelayStats=JSON.stringify({requestedMs:this.delayMs,actualMs:now-selected.capturedAt,presentedAtMs:now,queuedFrames:this.queue.length,presentedFrames:this.presented,skippedFrames:this.skipped});
    } finally {selected.frame.close();}
  }
  fail(error,epoch) {
    if(epoch!==this.epoch)return;
    this.stop();notice((this.video.getAttribute('aria-label')||'视频').replace('实时视频','')+'延时播放失败',true);
  }
  stop() {
    this.active=false;this.epoch++;
    if(this.frameRequest!==null)this.video.cancelVideoFrameCallback(this.frameRequest);
    if(this.captureRequest!==null)cancelAnimationFrame(this.captureRequest);
    if(this.paintRequest!==null)cancelAnimationFrame(this.paintRequest);
    this.frameRequest=this.captureRequest=this.paintRequest=null;
    for(const item of this.queue)item.frame.close();this.queue=[];this.inFlight=0;this.lastMediaTime=null;
    this.context.clearRect(0,0,this.canvas.width,this.canvas.height);
    delete this.video.dataset.presentationDelayStats;
  }
}
class Player {
  constructor(key) {this.key=key;this.video=$(key+'-video');this.delay=['original','h264'].includes(key)?new DelayedVideo(this.video,200):null;this.generation=0;this.pc=null;this.session=null;this.abort=null;this.retry=null;this.fps=null;this.sample=null;this.video.addEventListener('playing',()=>{if(this.pc){this.delay?.start();setState(key,'播放中',true);}});this.video.addEventListener('loadedmetadata',()=>this.resolution());}
  resolution() {$(this.key+'-res').textContent=this.video.videoWidth?`${this.video.videoWidth} × ${this.video.videoHeight} · ${this.fps===null?'—':this.fps.toFixed(0)} FPS`:'— × — · — FPS';}
  releaseSession(url) {if(url)fetch(url,{method:'DELETE',keepalive:true}).catch(()=>{});}
  close(show=true) {this.generation++;this.delay?.stop();clearTimeout(this.retry);this.retry=null;this.abort?.abort();this.abort=null;this.releaseSession(this.session);this.session=null;if(this.pc){this.pc.ontrack=null;this.pc.onconnectionstatechange=null;this.pc.close();this.pc=null;}this.video.srcObject=null;this.fps=null;this.sample=null;metric(this.key,{},false);this.resolution();if(show)setState(this.key,'已断开');}
  schedule() {if(!enabled||this.retry)return;this.retry=setTimeout(()=>{this.retry=null;this.connect();},3000);}
  async connect() {
    this.close(false);if(!enabled)return;const generation=this.generation;
    const path=pathFor(settings[this.key]);if(!path){setState(this.key,'地址无效');return;}
    setState(this.key,'连接中');const abort=new AbortController();this.abort=abort;const timer=setTimeout(()=>abort.abort(),8000);let pc;
    try {
      pc=new RTCPeerConnection({iceServers:[]});this.pc=pc;pc.addTransceiver('video',{direction:'recvonly'});
      pc.ontrack=e=>{if(generation!==this.generation)return;this.video.srcObject=e.streams[0]||new MediaStream([e.track]);this.video.play().catch(()=>setState(this.key,'播放失败'));};
      pc.onconnectionstatechange=()=>{if(generation!==this.generation)return;if(['failed','disconnected'].includes(pc.connectionState)){setState(this.key,'连接断开');this.schedule();}};
      await pc.setLocalDescription(await pc.createOffer());await iceReady(pc,abort.signal);
      if(generation!==this.generation)return;
      const url=settings.webrtc.replace(/\/$/,'')+'/'+path.split('/').map(encodeURIComponent).join('/')+'/whep';
      const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/sdp'},body:pc.localDescription.sdp,signal:abort.signal});
      if(!response.ok)throw new Error('WHEP '+response.status);
      const location=response.headers.get('Location');const session=location?new URL(location,url).href:null;
      if(generation!==this.generation){this.releaseSession(session);return;}this.session=session;
      const sdp=await response.text();if(generation!==this.generation)return;
      await pc.setRemoteDescription({type:'answer',sdp});
    } catch(error) {
      if(generation!==this.generation)return;
      this.close(false);setState(this.key,statusSnapshot?.feeds[this.key]?'连接失败':'未发布');this.schedule();
    } finally {clearTimeout(timer);}
  }
  async stats() {
    const pc=this.pc;if(!pc||pc.connectionState!=='connected')return;
    try {const reports=await pc.getStats();if(pc!==this.pc)return;reports.forEach(report=>{if(report.type==='inbound-rtp'&&(report.kind==='video'||report.mediaType==='video')){this.video.dataset.receptionStats=JSON.stringify({timestamp:report.timestamp,bytesReceived:report.bytesReceived,framesReceived:report.framesReceived,framesDecoded:report.framesDecoded,framesDropped:report.framesDropped,framesPerSecond:report.framesPerSecond,packetsLost:report.packetsLost,jitter:report.jitter,jitterBufferDelay:report.jitterBufferDelay,jitterBufferEmittedCount:report.jitterBufferEmittedCount,totalDecodeTime:report.totalDecodeTime,freezeCount:report.freezeCount});if(Number.isFinite(report.framesPerSecond))this.fps=report.framesPerSecond;else if(this.sample&&report.timestamp>this.sample.time)this.fps=Math.max(0,(report.framesDecoded-this.sample.frames)*1000/(report.timestamp-this.sample.time));this.sample={time:report.timestamp,frames:report.framesDecoded,bytes:report.bytesReceived};this.resolution();}});}catch{}
  }
}
for(const c of channels)players[c.key]=new Player(c.key);
function reconnect() {enabled=true;for(const p of Object.values(players))p.connect();}
const format=(number,digits=0)=>Number(number).toLocaleString('en-US',{minimumFractionDigits:digits,maximumFractionDigits:digits});
function metric(key, value, ready) {
  for(const [field,digits] of [['kbs',1],['bpp',4],['ratio',1]])$(key+'-'+field).textContent=ready&&Number.isFinite(value[field])?format(value[field],digits):'—';
}
// Store timestamped measurements; gaps are left empty rather than plotted as zero.
class BitrateHistory {
  constructor(windowMs=30000){this.windowMs=windowMs;this.samples=[];}
  add(time,values){this.samples.push({time,...values});this.prune(time);}
  prune(now){this.samples=this.samples.filter(sample=>sample.time>=now-this.windowMs);}
  segments(key,now){
    this.prune(now);const segments=[];let current=[];
    for(const sample of this.samples){
      if(!Number.isFinite(sample[key])||sample[key]<0||(current.length&&sample.time-current[current.length-1].time>2500)){if(current.length)segments.push(current);current=[];}
      if(Number.isFinite(sample[key])&&sample[key]>=0)current.push(sample);
    }
    if(current.length)segments.push(current);return segments;
  }
}
const bitrateHistory=new BitrateHistory();
function drawBitrate(now=performance.now()) {
  const chart=$('bitrate-chart'),width=Math.max(240,chart.clientWidth),height=chart.clientHeight||146;
  const left=48,right=12,top=12,bottom=24,w=width-left-right,h=height-top-bottom;
  bitrateHistory.prune(now);
  const peak=Math.max(1,...bitrateHistory.samples.flatMap(sample=>[sample.h264,sample.mlvc]).filter(Number.isFinite));
  const raw=peak*1.15,scale=10**Math.floor(Math.log10(raw)),upper=Math.ceil(raw/scale)*scale;
  const x=time=>left+w*(1-(now-time)/30000),y=value=>top+h*(1-value/upper);
  let markup='';
  for(const fraction of [0,.5,1]){const value=upper*fraction,position=y(value);markup+=`<line class="chart-grid" x1="${left}" x2="${width-right}" y1="${position}" y2="${position}"/><text x="${left-9}" y="${position+3}" text-anchor="end">${format(value,value<10?1:0)}</text>`;}
  for(const [fraction,label,anchor] of [[0,'30 秒前','start'],[.5,'15 秒前','middle'],[1,'现在','end']])markup+=`<text x="${left+w*fraction}" y="${height-4}" text-anchor="${anchor}">${label}</text>`;
  for(const key of ['h264','mlvc'])for(const segment of bitrateHistory.segments(key,now)){
    markup+=`<path class="chart-line ${key}-line" d="${segment.map((sample,index)=>`${index?'L':'M'}${x(sample.time).toFixed(2)},${y(sample[key]).toFixed(2)}`).join(' ')}"/>`;
    const last=segment[segment.length-1];markup+=`<circle class="${key}-dot" cx="${x(last.time).toFixed(2)}" cy="${y(last[key]).toFixed(2)}" r="2.5"/>`;
  }
  chart.setAttribute('viewBox',`0 0 ${width} ${height}`);chart.innerHTML=markup;
}
new ResizeObserver(()=>drawBitrate()).observe($('bitrate-chart'));
let lastError='';
async function refresh() {
  if(refreshing)return;refreshing=true;
  try {
    const status=await api('/api/status');statusSnapshot=status;
    for(const c of channels){const published=!!status.feeds[c.key];
      if(!published&&$(c.key+'-card').classList.contains('live')){players[c.key].close(false);setState(c.key,'未发布');if(enabled)players[c.key].schedule();}
      if(!players[c.key].pc&&!players[c.key].retry)setState(c.key,enabled?(published?'等待连接':'未发布'):'已断开');
      const ready=c.key==='original'||(c.key==='mlvc'?status.mlvc_actual:status.h264_actual);
      metric(c.key,status.metrics[c.key],ready);
      players[c.key].stats();
    }
    bitrateHistory.add(performance.now(),{h264:status.feeds.h264&&status.h264_actual?status.metrics.h264.kbs:null,mlvc:status.feeds.mlvc&&status.mlvc_actual?status.metrics.mlvc.kbs:null});drawBitrate();
    const error=status.error||status.mlvc_stats_error;if(error&&error!==lastError)notice(error,true);lastError=error;
  } catch(error) {if(lastError!=='offline')notice('控制服务连接失败',true);lastError='offline';for(const c of channels)metric(c.key,{},false);bitrateHistory.add(performance.now(),{h264:null,mlvc:null});drawBitrate();}
  finally {refreshing=false;}
}
async function pipeline(action) {
  $('start-button').disabled=$('stop-button').disabled=true;
  try {if(action==='stop'){enabled=false;for(const p of Object.values(players))p.close();}
    const result=await api('/api/'+action,action==='start'?{...settings,mode:'same_quality'}:{});notice(result.message);
    if(action==='start'){if(enabled)players.h264.connect();else reconnect();}await refresh();
  }catch(error){notice(error.message,true);if(action==='start'&&!enabled)reconnect();}
  finally{$('stop-button').disabled=false;$('start-button').disabled=false;}
}
$('start-button').addEventListener('click',()=>pipeline('start'));
$('stop-button').addEventListener('click',()=>pipeline('stop'));
$('reconnect-button').addEventListener('click',reconnect);
for(const button of document.querySelectorAll('[data-fullscreen]'))button.addEventListener('click',()=>$(button.dataset.fullscreen+'-wrap').requestFullscreen().catch(error=>notice('全屏失败：'+error.message,true)));
function fillSettings(){for(const [key,value] of Object.entries(settings))if($(key)&&$(key).matches('input,select')){let display=value;if(key==='mlvc'){const source=new URL(value);source.pathname='';source.search='';source.hash='';display=source.href;}$(key).value=display;}}
for(const button of document.querySelectorAll('[data-open-settings]'))button.addEventListener('click',()=>{fillSettings();$('settings-error').hidden=true;$('settings-dialog').showModal();});
for(const id of ['close-settings','cancel-settings'])$(id).addEventListener('click',()=>$('settings-dialog').close());
$('settings-dialog').addEventListener('close',fillSettings);
$('settings-form').addEventListener('submit',async e=>{e.preventDefault();const draft={...settings};for(const field of $('settings-form').querySelectorAll('input,select'))draft[field.name]=field.type==='number'?Number(field.value):field.value.trim();
  $('save-settings').disabled=true;
  try {const semanticSource=new URL(settings.mlvc),semanticServer=new URL(draft.mlvc);semanticServer.pathname=semanticSource.pathname;semanticServer.search=semanticSource.search;draft.mlvc=semanticServer.href;for(const key of ['original','h264','mlvc'])if(new URL(draft[key]).protocol!=='rtsp:')throw new Error('视频源地址必须使用 rtsp://');if(!['http:','https:'].includes(new URL(draft.webrtc).protocol))throw new Error('WebRTC 地址必须使用 http:// 或 https://');const applyQp=draft.mode==='same_quality'&&draft.h264_qp!==settings.h264_qp&&enabled;await api('/api/config',draft);settings=draft;$('settings-dialog').close();if(applyQp)await pipeline('start');else reconnect();notice('已保存');}
  catch(error){$('settings-error').hidden=false;$('settings-error').textContent=String(error.message).replace(/mlvc/gi,'语义压缩');}finally{$('save-settings').disabled=false;}
});
function clock(){const now=new Date();$('clock').textContent=now.toLocaleTimeString('zh-CN',{hour12:false});$('clock').dateTime=now.toISOString();}clock();setInterval(clock,1000);
window.addEventListener('pagehide',()=>{enabled=false;for(const p of Object.values(players))p.close(false);});
window.addEventListener('pageshow',e=>{if(e.persisted)reconnect();});
async function initialize(){try{settings={...await api('/api/config'),mode:'same_quality'};fillSettings();await refresh();reconnect();}catch(error){notice('读取配置失败：'+error.message,true);}}
initialize();setInterval(refresh,1000);
