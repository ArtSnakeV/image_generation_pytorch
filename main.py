import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import *
from tkinter import ttk

import tkinter
# tkinter._test()


model_id = "runwayml/stable-diffusion-v1-5"

# Тип середовища
device = "cuda" if torch.cuda.is_available() else "cpu"


# Створимо інструмент для генерації
# Пайплайн для генерації
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    # вимкнення перевірки NSFW, проти помилкових спрацювань
    safety_checker=None
)

photo_images = []

# Функція для генерації і відображення зображення
def show_image():
    # prompt_text = prompt_entry.get()
    # if not prompt_text:
    #     print("Please enter a prompt.")
    #     return
    # image = pipe(
    #     prompt_text,
    #     width=512,
    #     height=512,
    #     negative_prompt="",  # тут можна вказати, чого не повинно бути на зображенні
    #     num_images=2,
    # ).images
    # # Збереження зображення
    # image.save("prompt.png")
    # image.show()
    global photo_images
    prompt_text = prompt_entry.get()
    if not prompt_text:
        print("Please enter a prompt.")
        return
    # Генеруємо ображення
    # images = pipe(
    #     prompt_text,
    #     width=512,
    #     height=512,
    #     negative_prompt="",
    # ).images
    images = []
    for _ in range(2):
        img = pipe(
            prompt_text,
            width=512,
            height=512,
            negative_prompt=""
        ).images[0]
        images.append(img)
    # Перевіряємо кількість зображень
    print(f"Number of images generated: {len(images)}")
    # Зберігаємо наші зображення
    for i, img in enumerate(images):
        img.save(f"test_image_{i}.png")
    # Очищуємо віконечко від старих зображень
    for widget in image_frame.winfo_children():
        widget.destroy()
    #
    for img in images:
        img_tk = ImageTk.PhotoImage(img)
        photo_images.append(img_tk)
        label = tk.Label(image_frame, image=img_tk)
        label.pack(side=tk.LEFT, padx=5)

    # # Зберігаємо зображення
    # images[0].save("image1.png")
    # images[1].save("image2.png")


###################
# Створимо нове вікно
root = tk.Tk()
root.title("Text to image generator")

# Поле вводу для тексту
prompt_label  = tk.Label(root, text="Please enter text for image generation here.")
prompt_label .pack(pady=10)

prompt_entry = tk.Entry(root, width=50)
prompt_entry.pack(pady=5)

# pipe = pipe.to(device)

# Додамо кнопку для зчитування тексту і генерації зображення
button = tk.Button(root, text="Show Image", command=lambda: show_image())
button.pack(pady=10)


# Генерація зображення
# images = pipe(prompt).images[0]

# Додамо рамочку для зображень
image_frame = tk.Frame(root)
image_frame.pack(pady=10)

# Run the app
root.mainloop()




