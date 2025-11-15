import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

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

pipe = pipe.to(device)

# Промпт для генерації
prompt = """
Amazing clear water lake
"""

# Генерація зображення
# images = pipe(prompt).images[0]
image = pipe(
    prompt,
    width=512,
    height=512,
    negative_prompt="" # тут можна вказати, чого не повинно бути на зображенні
).images[0]

# Збереження зображення
image.save("prompt.png")
image.show()
