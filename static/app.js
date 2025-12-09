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
  const data = await res.json();
  if(data.error){ appendSystem(data.error); return; }
  appendMessage('Seller', data.seller_text, data.intent);
}

document.getElementById('start-btn').addEventListener('click', startScenario);
document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('message-input').addEventListener('keydown', (e)=>{ if(e.key==='Enter') sendMessage(); });
