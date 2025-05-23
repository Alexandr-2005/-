from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np
import torch

from neuralnet import load_model
import mnist_loader  # твой загрузчик

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Загрузка модели
net = load_model("trained_network.pkl")
net.eval()
device = next(net.parameters()).device

# Загрузка данных
_, _, test_data = mnist_loader.load_data_wrapper()
raw_test = list(test_data)  # test_data — список (img, label), img numpy (784,1)

def preprocess_image(pil_img):
    bbox = pil_img.getbbox()
    if not bbox:
        return None
    img = pil_img.crop(bbox).filter(ImageFilter.GaussianBlur(1))
    img = img.resize((20, 20), Image.NEAREST)
    new_img = Image.new('L', (28, 28), color=0)
    offset = ((28 - 20) // 2, (28 - 20) // 2)
    new_img.paste(img, offset)
    new_img = new_img.filter(ImageFilter.MaxFilter(3))
    return new_img

def image_to_array(pil_img):
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return arr.reshape((784,))

@socketio.on('connect')
def on_connect():
    emit('connected', {'test_count': len(raw_test)})

@socketio.on('recognize')
def on_recognize(data):
    mode = data.get('mode')
    proc = None

    if mode == 'draw':
        try:
            img_b64 = data.get('image', '')
            raw = base64.b64decode(img_b64.split(',')[1])
            pil = Image.open(BytesIO(raw)).convert('L')
        except Exception:
            return emit('error', {'msg': 'Неверный формат изображения.'})
        proc = preprocess_image(pil)
        if proc is None:
            return emit('error', {'msg': 'Нарисуйте цифру.'})
        arr = image_to_array(proc)

    elif mode == 'test':
        idx = int(data.get('index', 0))
        if not (0 <= idx < len(raw_test)):
            return emit('error', {'msg': 'Индекс вне диапазона.'})
        img, _ = raw_test[idx]  
        img_2d = img.reshape(28, 28) * 255
        pil = Image.fromarray(img_2d).convert('L')
        proc = preprocess_image(pil)
        if proc is None:
            return emit('error', {'msg': 'Ошибка обработки тестового изображения.'})
        arr = image_to_array(proc)

    else:
        return emit('error', {'msg': 'Неизвестный режим.'})

    tensor = torch.from_numpy(arr).float().to(device).unsqueeze(0)  # [1,784]
    with torch.no_grad():
        outputs = net(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

    pred = int(probs.argmax())
    top3_idx = probs.argsort()[-3:][::-1]
    probs_dict = {str(i): float(probs[i]) for i in top3_idx}

    preview_b64 = ''
    if proc is not None:
        buf = BytesIO()
        proc.resize((100, 100), Image.NEAREST).save(buf, format='PNG')
        preview_b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

    emit('result', {
        'prediction': pred,
        'probabilities': probs_dict,
        'preview': preview_b64
    })

@socketio.on('preview')
def on_preview(data):
    try:
        img_b64 = data.get('image', '')
        raw = base64.b64decode(img_b64.split(',')[1])
        pil = Image.open(BytesIO(raw)).convert('L')
        proc = preprocess_image(pil)
        if proc is None:    
            return  
        buf = BytesIO()
        proc.resize((100, 100), Image.NEAREST).save(buf, format='PNG')
        preview_b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        emit('preview', {'preview': preview_b64})
    except Exception:
        pass  

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
