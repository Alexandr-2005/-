document.addEventListener('DOMContentLoaded', () => {
  const socket = io();
  const canvas = document.getElementById('draw-canvas');
  const ctx = canvas.getContext('2d');

  let drawing = false;
  let brushSize = 8;
  let previewTimeout = null;

  // Настройка canvas
  canvas.style.background = 'black';
  ctx.strokeStyle = 'white';
  ctx.lineWidth = brushSize;
  ctx.lineCap = 'round';

  canvas.onmousedown = () => { drawing = true; ctx.beginPath(); };
  canvas.onmouseup   = () => {
    drawing = false;
    if (previewTimeout) {
      clearTimeout(previewTimeout);
      previewTimeout = null;
    }
  };
  canvas.onmousemove = e => {
    if (!drawing) return;
  
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
  
    // Отправка превью сразу после каждого движения
    const dataURL = canvas.toDataURL('image/png');
    socket.emit('preview', { image: dataURL });
  };
  

  // Управление кистью
  document.getElementById('brush-size').oninput = e => {
    brushSize = +e.target.value;
    ctx.lineWidth = brushSize;
  };

  // Кнопка Clear
  document.getElementById('clear-btn').onclick = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    document.getElementById('result').textContent = 'Result:';
    document.getElementById('probs').textContent = '';
    document.getElementById('preview-img').src = '';
  };

  // Слайдер test-idx
  document.getElementById('test-idx').oninput = e => {
    document.getElementById('idx-val').textContent = e.target.value;
  };

  // Load test sample
  document.getElementById('load-test-btn').onclick = () => {
    const idx = document.getElementById('test-idx').value;
    socket.emit('recognize', { mode: 'test', index: idx });
  };

  // Recognize button
  document.getElementById('recognize-btn').onclick = () => {
    const dataURL = canvas.toDataURL('image/png');
    socket.emit('recognize', { mode: 'draw', image: dataURL });
  };

  // При подключении получаем test_count
  socket.on('connected', data => {
    const slider = document.getElementById('test-idx');
    slider.max = data.test_count - 1;
    document.getElementById('idx-val').textContent = slider.value;
  });

  // Ошибки
  socket.on('error', data => {
    alert(data.msg);
  });

  // Пришёл preview
  socket.on('preview', data => {
    document.getElementById('preview-img').src = data.preview;
  });

  // Итоговое предсказание
  socket.on('result', data => {
    document.getElementById('result').textContent = `Prediction: ${data.prediction}`;
    document.getElementById('probs').textContent =
      Object.entries(data.probabilities)
            .map(([d,p])=>`${d}: ${(p*100).toFixed(1)}%`).join('  ');
    if (data.preview) {
      document.getElementById('preview-img').src = data.preview;
    }
  });
});
