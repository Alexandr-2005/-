import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageChops, ImageFilter, ImageTk
import numpy as np
import threading
from neuralnet import load_model
import mnist_loader

class DigitRecognizer:
    def __init__(self, model_path="trained_network.pkl"):
        self.window = tk.Tk()
        self.window.title("EMNIST Digit Recognizer")

        try:
            self.net = load_model(model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load network: {e}")
            self.window.destroy()
            return
            
        # Загрузка тестовых данных
        _, _, test_data = mnist_loader.load_data_wrapper()
        self.test_data = list(test_data)
        self.num_tests = len(self.test_data)

        # Настройки интерфейса
        self.canvas_size = 280
        self.stroke_width = 8
        
        # Основной холст для рисования
        self.canvas = tk.Canvas(self.window, width=self.canvas_size,
                               height=self.canvas_size, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=4)
        
        # Изображение для обработки
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Переменные для рисования
        self.last_x = None
        self.last_y = None
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # Кнопки управления
        self.btn_recognize = tk.Button(self.window, text="Recognize", command=self.start_recognition)
        self.btn_clear = tk.Button(self.window, text="Clear", command=self.clear_canvas)
        self.btn_load = tk.Button(self.window, text="Load Test Sample", command=self.load_test_sample)
        self.btn_recognize.grid(row=1, column=0, pady=5)
        self.btn_clear.grid(row=1, column=1)
        self.btn_load.grid(row=1, column=2)

        # Слайдер для выбора тестового примера
        tk.Label(self.window, text="Test idx:").grid(row=2, column=0)
        self.test_idx = tk.Scale(self.window, from_=0, to=self.num_tests-1,
                               orient='horizontal', length=200)
        self.test_idx.grid(row=2, column=1, columnspan=2)

        # Настройки кисти и порога
        tk.Label(self.window, text="Brush size:").grid(row=3, column=0)
        self.brush_scale = tk.Scale(self.window, from_=2, to=30, 
                                   orient='horizontal', command=self.change_brush)
        self.brush_scale.set(self.stroke_width)
        self.brush_scale.grid(row=3, column=1)
        
        tk.Label(self.window, text="Threshold:").grid(row=3, column=2)
        self.threshold = tk.Scale(self.window, from_=0, to=255, orient='horizontal')
        self.threshold.set(128)
        self.threshold.grid(row=3, column=3)

        # Метки для вывода результатов
        self.label = tk.Label(self.window, text="Result: ", font=('Arial', 24))
        self.label.grid(row=4, column=0, columnspan=4)
        self.prob_label = tk.Label(self.window, text="", font=('Arial', 14))
        self.prob_label.grid(row=5, column=0, columnspan=4)

        # Превью этапов обработки
        tk.Label(self.window, text="Centered").grid(row=6, column=0, columnspan=2)
        self.centered_canvas = tk.Canvas(self.window, width=100, height=100, bg='black')
        self.centered_canvas.grid(row=7, column=0, columnspan=2)
        
        tk.Label(self.window, text="Final").grid(row=6, column=2, columnspan=2)
        self.final_canvas = tk.Canvas(self.window, width=100, height=100, bg='black')
        self.final_canvas.grid(row=7, column=2, columnspan=2)

        self.preview_images = []  # Для хранения ссылок на изображения
        self.is_drawing = False
        self.update_preview_id = None

        self.window.mainloop()

    def change_brush(self, val):
        self.stroke_width = int(val)

    def draw_line(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  fill='white', width=self.stroke_width)
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                          fill=255, width=self.stroke_width)
        self.last_x, self.last_y = event.x, event.y
        self.is_drawing = True
        if self.update_preview_id is None:
            self.update_preview_id = self.window.after(200, self.update_preview)

    def reset(self, event):
        self.last_x, self.last_y = None, None
        self.is_drawing = False
        if self.update_preview_id is not None:
            self.window.after_cancel(self.update_preview_id)
            self.update_preview_id = None

    def clear_canvas(self):
        self.canvas.delete('all')
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=0)
        self.label.config(text="Result: ")
        self.prob_label.config(text="")
        self.centered_canvas.delete('all')
        self.final_canvas.delete('all')

    def load_test_sample(self):
        idx = self.test_idx.get()
        x, y = self.test_data[idx]
        img = Image.fromarray((x.reshape(28,28)*255).astype(np.uint8))
        
        # Поворот для EMNIST Digits
        img = img.rotate(90, expand=True)
        
        img = img.resize((self.canvas_size, self.canvas_size), Image.NEAREST)
        self.image.paste(img)
        self.canvas_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor='nw', image=self.canvas_img)
        self.label.config(text=f"True: {np.argmax(y) if hasattr(y, 'shape') else int(y)}")
        self.prob_label.config(text="")

    def preprocess(self, for_preview=False):
        bbox = self.image.getbbox()
        if bbox is None:
            return None, None, None
            
        # Обрезаем и обрабатываем изображение
        img = self.image.crop(bbox)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.resize((20, 20), resample=Image.NEAREST)
        
        # Создаем новое изображение 28x28
        new_img = Image.new('L', (28, 28), color=0)
        offset = ((28 - 20) // 2, (28 - 20) // 2)
        new_img.paste(img, offset)
        new_img = new_img.filter(ImageFilter.MaxFilter(size=3))
        
        # Поворачиваем для EMNIST
        new_img = new_img.rotate(-90)
        centered_img = new_img.copy()

        # Бинаризация и центрирование
        arr = np.array(new_img)
        th = self.threshold.get()
        arr = (arr > th).astype(np.uint8)
        coords = np.column_stack(np.where(arr > 0))
        
        if coords.size:
            cy, cx = coords.mean(axis=0)
            shift_x = int(round((arr.shape[1]/2)-cx))
            shift_y = int(round((arr.shape[0]/2)-cy))
            new_img = ImageChops.offset(new_img, shift_x, shift_y)
            arr = (np.array(new_img) > th).astype(np.float32)
            
        final_img = new_img

        if for_preview:
            return centered_img, final_img
        else:
            return arr.astype(np.float32).reshape((784, 1)), centered_img, final_img

    def update_preview(self):
        if not self.is_drawing:
            return
        centered_img, final_img = self.preprocess(for_preview=True)
        if centered_img is not None and final_img is not None:
            self.display_preview(centered_img, self.centered_canvas)
            self.display_preview(final_img, self.final_canvas)
        self.update_preview_id = self.window.after(200, self.update_preview)

    def display_preview(self, img, canvas):
        img_preview = img.resize((100, 100), Image.NEAREST)
        preview_image = ImageTk.PhotoImage(img_preview)
        self.preview_images.append(preview_image)
        canvas.delete('all')
        canvas.create_image(0, 0, anchor='nw', image=preview_image)

    def start_recognition(self):
        threading.Thread(target=self.recognize).start()

    def recognize(self):
        processed, centered_img, final_img = self.preprocess()
        if processed is None:
            messagebox.showinfo("Info", "Please draw a digit or load a test sample.")
            return
            
        self.display_preview(centered_img, self.centered_canvas)
        self.display_preview(final_img, self.final_canvas)
        
        output = self.net.feedforward(processed)
        probs = output.flatten()
        pred = np.argmax(probs)
        top3 = probs.argsort()[-3:][::-1]
        prob_text = '  '.join(f"{i}:{probs[i]*100:.1f}%" for i in top3)
        self.label.config(text=f"Prediction: {pred} ({probs[pred]*100:.1f}%)")
        self.prob_label.config(text=prob_text)

if __name__ == "__main__":
    DigitRecognizer()