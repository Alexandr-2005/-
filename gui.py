import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageChops, ImageFilter, ImageTk
import numpy as np
import threading
import torch

from neuralnet import load_model
import mnist_loader

class DigitRecognizer:
    def __init__(self, model_path="trained_network.pkl"):
        self.window = tk.Tk()
        self.window.title("Digit Recognizer")
        layer_sizes = [28*28, 512, 256, 128, 10]

        try:
            self.net = load_model(model_path, layer_sizes)
            self.net.eval()
            self.device = next(self.net.parameters()).device
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load network: {e}")
            self.window.destroy()
            return

        _, _, test_data = mnist_loader.load_data_wrapper()
        self.raw_test = list(test_data)

        # Инициализация интерфейса
        self.canvas_size = 280
        self.stroke_width = 8
        self.canvas = tk.Canvas(self.window, width=self.canvas_size,
                                height=self.canvas_size, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # Элементы управления
        tk.Button(self.window, text="Recognize", command=self.start_recognition).grid(row=1, column=0)
        tk.Button(self.window, text="Clear", command=self.clear_canvas).grid(row=1, column=1)
        tk.Button(self.window, text="Load Test Sample", command=self.load_test_sample).grid(row=1, column=2)

        # Настройки
        tk.Label(self.window, text="Test idx:").grid(row=2, column=0)
        self.test_idx = tk.Scale(self.window, from_=0, to=len(self.raw_test)-1,
                                 orient='horizontal', length=200)
        self.test_idx.grid(row=2, column=1, columnspan=2)

        # Кисть
        tk.Label(self.window, text="Brush size:").grid(row=3, column=0)
        self.brush_scale = tk.Scale(self.window, from_=2, to=30,
                                    orient='horizontal', command=self.change_brush)
        self.brush_scale.set(self.stroke_width)
        self.brush_scale.grid(row=3, column=1)

        # Результаты
        self.label = tk.Label(self.window, text="Result: ", font=('Arial', 24))
        self.label.grid(row=4, column=0, columnspan=4)
        self.prob_label = tk.Label(self.window, text="", font=('Arial', 14))
        self.prob_label.grid(row=5, column=0, columnspan=4)

        # Превью изображений
        tk.Label(self.window, text="Processed").grid(row=6, column=0)
        self.processed_canvas = tk.Canvas(self.window, width=100, height=100, bg='black')
        self.processed_canvas.grid(row=7, column=0, columnspan=2)

        self.last_x = self.last_y = None
        self.is_drawing = False
        self.update_preview_id = None
        self.preview_images = []

        self.window.mainloop()

    def change_brush(self, val):
        self.stroke_width = int(val)

    def draw_line(self, event):
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill='white', width=self.stroke_width)
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill=255, width=self.stroke_width)
        self.last_x, self.last_y = event.x, event.y
        self.is_drawing = True
        if not self.update_preview_id:
            self.update_preview_id = self.window.after(200, self.update_preview)

    def reset(self, event):
        self.last_x = self.last_y = None
        self.is_drawing = False
        if self.update_preview_id:
            self.window.after_cancel(self.update_preview_id)
            self.update_preview_id = None

    def clear_canvas(self):
        self.canvas.delete('all')
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=0)
        self.label.config(text="Result: ")
        self.prob_label.config(text="")
        self.processed_canvas.delete('all')

    def load_test_sample(self):
        idx = self.test_idx.get()
        x, y = self.raw_test[idx]
        arr = (x.reshape(28, 28) * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        pil = pil.resize((self.canvas_size, self.canvas_size), Image.NEAREST)
        self.image.paste(pil)
        self.canvas_img = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, anchor='nw', image=self.canvas_img)
        self.label.config(text=f"True: {int(y)}")
        self.recognize(use_raw=True)

    def preprocess_image(self, pil_img):
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

    def get_processed_array(self, pil_img):
        new_img = self.preprocess_image(pil_img)
        if new_img is None:
            return None
        arr = np.array(new_img, dtype=np.float32) / 255.0
        return arr.reshape((784,))

    def update_preview(self):
        if not self.is_drawing:
            return
        processed_img = self.preprocess_image(self.image)
        if processed_img:
            self.display_preview(processed_img, self.processed_canvas)
        self.update_preview_id = self.window.after(200, self.update_preview)

    def display_preview(self, img, canvas):
        pv = img.resize((100, 100), Image.NEAREST)
        tkimg = ImageTk.PhotoImage(pv)
        self.preview_images.append(tkimg)
        canvas.delete('all')
        canvas.create_image(0, 0, anchor='nw', image=tkimg)

    def start_recognition(self):
        threading.Thread(target=self.recognize).start()

    def recognize(self, use_raw=False):
        if use_raw:
            idx = self.test_idx.get()
            x, y = self.raw_test[idx]
            processed = x.reshape((1, 784))
            true_label = int(y)
        else:
            arr = self.get_processed_array(self.image)
            if arr is None:
                messagebox.showinfo("Info", "Draw or load a test sample.")
                return
            processed = arr.reshape((1, 784))
            true_label = None

        tensor = torch.from_numpy(processed).float().to(self.device)

        with torch.no_grad():
            outputs = self.net(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

        pred = int(probs.argmax())
        top3 = probs.argsort()[-3:][::-1]
        prob_text = '  '.join(f"{i}:{probs[i]*100:.1f}%" for i in top3)

        if use_raw:
            self.label.config(text=f"True: {true_label} | Pred: {pred} ({probs[pred]*100:.1f}%)")
        else:
            self.label.config(text=f"Prediction: {pred} ({probs[pred]*100:.1f}%)")

        self.prob_label.config(text=prob_text)

if __name__ == "__main__":
    DigitRecognizer()
