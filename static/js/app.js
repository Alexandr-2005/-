const socket = io();
window.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('drawCanvas');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'black'; ctx.fillRect(0,0,280,280);
  ctx.strokeStyle = 'white'; ctx.lineWidth = 8;
  let drawing = false;
  let recognitionTimer = null;

  const brush = document.getElementById('brushSize');
  const threshold = document.getElementById('threshold');
  brush.oninput = () => ctx.lineWidth = brush.value;

  const sendRecognition = () => {
    const dataURL = canvas.toDataURL();
    socket.emit('recognize', { image: dataURL, threshold: parseInt(threshold.value) });
  };

  canvas.onmousedown = () => {
    drawing = true; ctx.beginPath();
  };
  canvas.onmouseup = () => {
    drawing = false;
    sendRecognition();
  };
  canvas.onmouseout = () => drawing = false;
  canvas.onmousemove = e => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX-rect.left, e.clientY-rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX-rect.left, e.clientY-rect.top);
    // Отправляем распознавание через 500ms после последнего движения
    if (recognitionTimer) clearTimeout(recognitionTimer);
    recognitionTimer = setTimeout(sendRecognition, 500);
  };

  document.getElementById('btnClear').onclick = () => {
    ctx.fillRect(0,0,280,280);
    document.getElementById('prediction').textContent = 'Result:';
    document.getElementById('probabilities').textContent = '';
    document.getElementById('preview').src = '';
    if (recognitionTimer) clearTimeout(recognitionTimer);
  };

  document.getElementById('btnRecognize').onclick = sendRecognition;

  socket.on('result', data => {
    if (data.error) { alert(data.error); return; }
    document.getElementById('prediction').textContent =
      `Prediction: ${data.prediction} (${(data.confidence*100).toFixed(1)}%)`;
    document.getElementById('probabilities').textContent =
      data.top3.map(o => `${o.digit}:${(o.prob*100).toFixed(1)}%`).join('  ');
    document.getElementById('preview').src = data.preview;
  });

  document.getElementById('btnLoad').onclick = () => {
    const idx = document.getElementById('testIdx').value;
    socket.emit('load_test', { idx });
  };

  socket.on('test_loaded', data => {
    const img = new Image();
    img.onload = () => ctx.drawImage(img,0,0);
    img.src = data.image;
    document.getElementById('prediction').textContent = `True: ${data.label}`;
    document.getElementById('probabilities').textContent = '';
  });
});