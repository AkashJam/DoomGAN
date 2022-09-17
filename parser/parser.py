from WADEditor import WADReader
from matplotlib import pyplot as plt

reader = WADReader()

wad_with_features = reader.extract("../dataset/scraped/doom/0/0.WAD")
wad = wad_with_features["wad"]
levels = wad_with_features["levels"]
features = levels[0]["features"]
maps = levels[0]["maps"]

print(wad['directory'])
# plt.imshow(maps["thingsmap"])
# plt.show()

