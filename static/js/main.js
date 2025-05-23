document.addEventListener('DOMContentLoaded', () => {
  const socket = io();
  const canvas = document.getElementById('draw-canvas');
  const ctx = canvas.getContext('2d');

  // Инициализация canvas
  let drawing = false;
  let brushSize = 8;
  
  // Увеличиваем разрешение canvas
  canvas.width = 280;
  canvas.height = 280;
  canvas.style.width = '280px';
  canvas.style.height = '280px';
  
  // Настройки рисования
  canvas.style.background = 'black';
  ctx.strokeStyle = 'white';
  ctx.lineWidth = brushSize;
  ctx.lineCap = 'round';
  ctx.imageSmoothingEnabled = false;

  let lastPreviewTime = 0;
  const previewInterval = 50;

  // Обработчики событий мыши
  canvas.onmousedown = (e) => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
  };

  canvas.onmouseup = () => drawing = false;
  canvas.onmouseout = () => drawing = false;

  canvas.onmousemove = e => {
    if (!drawing) return;
    
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    
    // Отправка превью
    const now = Date.now();
    if (now - lastPreviewTime > previewInterval) {
      lastPreviewTime = now;
      const dataURL = canvas.toDataURL('image/png');
      socket.emit('preview', { image: dataURL });
    }
  };

  // Элементы управления
  document.getElementById('brush-size').oninput = e => {
    brushSize = +e.target.value;
    ctx.lineWidth = brushSize;
  };

  document.getElementById('clear-btn').onclick = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').textContent = 'Result:';
    document.getElementById('probs').textContent = '';
    document.getElementById('preview-img').src = '';
  };

  document.getElementById('test-idx').oninput = e => {
    document.getElementById('idx-val').textContent = e.target.value;
  };

  document.getElementById('load-test-btn').onclick = () => {
    const idx = document.getElementById('test-idx').value;
    socket.emit('recognize', { mode: 'test', index: idx });
  };

  document.getElementById('recognize-btn').onclick = () => {
    const dataURL = canvas.toDataURL('image/png');
    socket.emit('recognize', { mode: 'draw', image: dataURL });
  };

  // WebSocket handlers
  socket.on('connected', data => {
    const slider = document.getElementById('test-idx');
    slider.max = data.test_count - 1;
    document.getElementById('idx-val').textContent = slider.value;
  });

  socket.on('error', data => alert(data.msg));

  socket.on('preview', data => {
    document.getElementById('preview-img').src = data.preview;
  });

  socket.on('result', data => {
    document.getElementById('result').textContent = `Prediction: ${data.prediction}`;
    document.getElementById('probs').textContent = 
      Object.entries(data.probabilities)
            .map(([d, p]) => `${d}: ${(p * 100).toFixed(1)}%`)
            .join('  ');
    if (data.preview) {
      document.getElementById('preview-img').src = data.preview;
    }
  });
});
