from PIL import Image

img = Image.open("img/tile2.jpg")
cols = 3
rows = 3

W, H = img.size

grid = Image.new("RGB", (W * cols, H * rows))

for y in range(rows):
    for x in range(cols):
        grid.paste(img, (x * W, y * H))

grid.save("results/tile_test4.png")