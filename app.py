from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from neuralnet import load_model
import mnist_loader
from PIL import Image, ImageChops, ImageFilter
import numpy as np
import io, base64

app = Flask(__name__)
socketio = SocketIO(app)

# Загрузка модели и данных
net = load_model("trained_network.pkl")
_, _, test_data = mnist_loader.load_data_wrapper()
test_data = list(test_data)

# Препроцессинг (по аналогии с gui.py)
def preprocess_image(img: Image.Image, threshold: int):
    bbox = img.getbbox()
    if bbox is None:
        return None, None
    img = img.crop(bbox).filter(ImageFilter.GaussianBlur(1)).resize((20,20), Image.NEAREST)
    new_img = Image.new('L', (28,28), 0)
    new_img.paste(img, ((28-20)//2, (28-20)//2))
    new_img = new_img.filter(ImageFilter.MaxFilter(3)).rotate(90)
    arr = (np.array(new_img) > threshold).astype(np.float32)
    coords = np.column_stack(np.where(arr>0))
    if coords.size:
        cy, cx = coords.mean(axis=0)
        shift_x = int(round(arr.shape[1]/2 - cx))
        shift_y = int(round(arr.shape[0]/2 - cy))
        new_img = ImageChops.offset(new_img, shift_x, shift_y)
        arr = (np.array(new_img) > threshold).astype(np.float32)
    return arr.reshape((784,1)), new_img

@app.route('/')
def index():
    return render_template('index.html', num_tests=len(test_data))

@socketio.on('recognize')
def handle_recognize(data):
    img_data = data['image'].split(',')[1]
    threshold = data.get('threshold', 128)
    img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('L')
    processed, final_img = preprocess_image(img, threshold)
    if processed is None:
        emit('result', {'error': 'Please draw or load an image.'})
        return
    # Предсказание
    probs = net.feedforward(processed).flatten()
    pred = int(np.argmax(probs))
    top3 = probs.argsort()[-3:][::-1]
    result = {
        'prediction': pred,
        'confidence': float(probs[pred]),
        'top3': [{ 'digit': int(i), 'prob': float(probs[i]) } for i in top3]
    }
    # Preview
    buf = io.BytesIO()
    final_img.resize((100,100), Image.NEAREST).save(buf, 'PNG')
    result['preview'] = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    emit('result', result)

@socketio.on('load_test')
def handle_load_test(data):
    idx = int(data['idx'])
    x, y = test_data[idx]
    img = Image.fromarray((x.reshape(28,28)*255).astype(np.uint8))
    img = img.rotate(90, expand=True).resize((280,280), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    emit('test_loaded', {
        'image': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode(),
        'label': int(np.argmax(y) if hasattr(y, 'shape') else y)
    })

if __name__ == '__main__':
    socketio.run(app, debug=True)