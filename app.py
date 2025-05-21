from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np
from neuralnet import load_model
import mnist_loader

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Загрузка сети и тестовых данных
net = load_model("trained_network.pkl")
_, _, test_data = mnist_loader.load_data_wrapper()
raw_test = list(test_data)


def preprocess_image(pil_img):
    bbox = pil_img.getbbox()
    if not bbox:
        return None
    img = pil_img.crop(bbox).filter(ImageFilter.GaussianBlur(1))
    img = img.resize((20, 20), Image.NEAREST)
    new_img = Image.new('L', (28, 28), color=0)
    offset = ((28 - 20) // 2, (28 - 20) // 2)
    new_img.paste(img, offset)
    new_img = new_img.filter(ImageFilter.MaxFilter(3)).rotate(-90)
    return new_img


def image_to_array(pil_img):
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return arr.reshape((784, 1))


@socketio.on('connect')
def on_connect():
    emit('connected', {'test_count': len(raw_test)})


@socketio.on('recognize')
def on_recognize(data):
    mode = data.get('mode')
    proc = None

    if mode == 'draw':
        # Получаем Base64-поле и конвертируем
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
        x, _ = raw_test[idx]
        arr = x.reshape((784, 1))

    else:
        return emit('error', {'msg': 'Неизвестный режим.'})

    # Предсказание
    output = net.feedforward(arr).flatten()
    pred = int(np.argmax(output))
    top3 = output.argsort()[-3:][::-1]
    probs = {str(i): float(output[i]) for i in top3}

    # Если есть preprocessed‐image, генерим preview
    preview_b64 = ''
    if proc is not None:
        buf = BytesIO()
        proc.resize((100, 100), Image.NEAREST).save(buf, format='PNG')
        preview_b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

    emit('result', {
        'prediction': pred,
        'probabilities': probs,
        'preview': preview_b64
    })


@socketio.on('preview')
def on_preview(data):
    # Приходит кадр, только предобработка
    try:
        img_b64 = data.get('image', '')
        raw = base64.b64decode(img_b64.split(',')[1])
        pil = Image.open(BytesIO(raw)).convert('L')
    except Exception:
        return emit('error', {'msg': 'Некорректный формат для превью.'})
    proc = preprocess_image(pil)
    if proc is None:
        return emit('error', {'msg': 'Нарисуйте цифру.'})
    buf = BytesIO()
    proc.resize((100, 100), Image.NEAREST).save(buf, format='PNG')
    preview_b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    emit('preview', {'preview': preview_b64})


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
