let convId = null;

function el(q){return document.querySelector(q)}

async function startScenario(){
  const form = document.getElementById('scenario-form');
  const fd = new FormData(form);
  const scenario = {};
  for(const [k,v] of fd.entries()) scenario[k]=v;

  const res = await fetch('/api/start', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({scenario})
  });
  const data = await res.json();
  convId = data.conv_id;
  el('#scenario-meta').textContent = data.scenario_meta;
  el('#messages').innerHTML = '';
  appendSystem('Scenario started. You are the BUYER. Type your offer.')
}

function appendMessage(role, text, intent){
  const m = document.createElement('div');
  m.className = 'msg ' + (role==='Buyer' ? 'user' : 'bot');
  m.innerHTML = `<div class="role">${role}</div><div class="text">${text}</div>`;
  if(intent){
    const it = document.createElement('div'); it.className='intent'; it.textContent = `Intent: ${intent}`;
    m.appendChild(it);
  }
  el('#messages').appendChild(m);
  el('#messages').scrollTop = el('#messages').scrollHeight;
}

function appendSystem(text){
  const m = document.createElement('div'); m.className='system'; m.textContent=text;
  el('#messages').appendChild(m);
}

async function sendMessage(){
  const input = el('#message-input');
  const txt = input.value.trim();
  if(!txt || !convId) return;
  appendMessage('Buyer', txt);
  input.value='';

  const res = await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conv_id:convId, message:txt})});
  const data = await res.json().catch(()=>null);
  if(!res.ok){ appendSystem(data?.error || 'Server error during generation'); return; }
  appendMessage('Seller', data.seller_text, data.intent);
  // If server returned Piper audio, play it; otherwise fall back to browser TTS
  if(data.audio_b64){
    const audio = new Audio('data:audio/wav;base64,'+data.audio_b64);
    audio.play().catch(()=>{});
  }else if(window.ttsEnabled){
    speakText(data.seller_text);
  }
}

document.getElementById('start-btn').addEventListener('click', startScenario);
document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('message-input').addEventListener('keydown', (e)=>{ if(e.key==='Enter') sendMessage(); });

// Client-side audio recording -> upload to server for Whisper transcription
const micBtn = document.getElementById('mic-btn');
let mediaRecorder = null;
let audioChunks = [];

async function startRecording(){
  if(!convId){ appendSystem('Start a scenario first.'); return; }
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    appendSystem('Browser does not support microphone recording.');
    return;
  }

  try{
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstart = () => { micBtn.classList.add('listening'); micBtn.textContent = '●'; };
    mediaRecorder.onstop = async () => {
      micBtn.classList.remove('listening'); micBtn.textContent = '🎤';
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      // send to server as multipart/form-data
      const fd = new FormData();
      fd.append('conv_id', convId);
      fd.append('audio', blob, 'recording.webm');
      appendSystem('Uploading audio for transcription...');
      try{
        const resp = await fetch('/api/message', { method: 'POST', body: fd });
        const data = await resp.json().catch(()=>null);
        if(!resp.ok){ appendSystem(data?.error || 'Server error during transcription/generation'); return; }
        // extract buyer message from returned history if present
        let buyerText = '';
        if(Array.isArray(data.history)){
          for(let i=data.history.length-1;i>=0;i--){ if(data.history[i][0]==='Buyer'){ buyerText = data.history[i][1]; break; } }
        }
        if(buyerText) appendMessage('Buyer', buyerText);
        appendMessage('Seller', data.seller_text, data.intent);
        if(data.audio_b64){
          const audio = new Audio('data:audio/wav;base64,'+data.audio_b64);
          audio.play().catch(()=>{});
        }else if(window.ttsEnabled){ speakText(data.seller_text); }
      }catch(err){ appendSystem('Server error: '+err); }
    };
    mediaRecorder.start();
  }catch(e){ console.error(e); appendSystem('Unable to access microphone: ' + e.message); }
}

// Click toggles recording
micBtn.addEventListener('click', ()=>{
  if(mediaRecorder && mediaRecorder.state==='recording'){
    mediaRecorder.stop();
  }else{
    startRecording();
  }
});

// Text-to-Speech: browser SpeechSynthesis
window.ttsEnabled = true;
function speakText(text){
  if(!('speechSynthesis' in window)) return;
  try{
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'en-US';
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }catch(e){ console.warn('TTS error', e); }
}

// TTS toggle button
const ttsBtn = document.getElementById('tts-btn');
if(ttsBtn){
  const update = ()=>{ ttsBtn.style.opacity = window.ttsEnabled ? '1' : '0.45'; };
  ttsBtn.addEventListener('click', ()=>{ window.ttsEnabled = !window.ttsEnabled; update(); });
  update();
}
