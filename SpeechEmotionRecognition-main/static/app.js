const form = document.getElementById('uploadForm');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const lstmEl = document.getElementById('lstm');
const cnnEl = document.getElementById('cnn');
const lstmConfEl = document.getElementById('lstmConf');
const cnnConfEl = document.getElementById('cnnConf');
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const preview = document.getElementById('preview');
let mediaRecorder;
let recordedChunks = [];
let lastObjectUrl;

async function submitAndPredict(file) {
  resultsEl.classList.add('hidden');
  statusEl.classList.remove('hidden');
  statusEl.textContent = 'Uploading and predicting...';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || 'Prediction failed');

    lstmEl.textContent = data.predictions.lstm_prediction;
    cnnEl.textContent = data.predictions.cnn_prediction;
    const lstmConf = data.predictions.lstm_confidence;
    const cnnConf = data.predictions.cnn_confidence;
    lstmConfEl.textContent = lstmConf != null ? `(${(lstmConf*100).toFixed(1)}%)` : '';
    cnnConfEl.textContent = cnnConf != null ? `(${(cnnConf*100).toFixed(1)}%)` : '';
    resultsEl.classList.remove('hidden');
    statusEl.classList.add('hidden');
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
}

// Auto submit on file select
document.getElementById('audioFile').addEventListener('change', (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) {
    submitAndPredict(file);
  }
});

// Keep manual submit working too
form.addEventListener('submit', (e) => {
  e.preventDefault();
  const file = document.getElementById('audioFile').files[0];
  if (file) submitAndPredict(file);
});

// Microphone recording using MediaRecorder
if (recordBtn && stopBtn) {
  recordBtn.addEventListener('click', async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recordedChunks.push(e.data);
      };
      mediaRecorder.onstop = async () => {
        const blob = new Blob(recordedChunks, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        preview.src = url;
        preview.classList.remove('hidden');
        // Auto-send for prediction
        const file = new File([blob], 'recording.webm', { type: 'audio/webm' });
        await submitAndPredict(file);
      };
      mediaRecorder.start();
      recordBtn.setAttribute('disabled', '');
      stopBtn.removeAttribute('disabled');
    } catch (err) {
      statusEl.classList.remove('hidden');
      statusEl.textContent = `Mic error: ${err.message}`;
    }
  });

  stopBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      recordBtn.removeAttribute('disabled');
      stopBtn.setAttribute('disabled', '');
    }
  });
}


