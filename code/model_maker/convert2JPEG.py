import PIL.Image
import glob
import os

lst_imgs = [i for i in glob.glob("./data/train/*.png")]

for i in lst_imgs:
    img = PIL.Image.open(i)
    img = img.convert("RGB")
    img.save(i, "JPEG")

print("Done.")