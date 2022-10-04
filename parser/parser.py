from WADEditor import WADReader
from matplotlib import pyplot as plt

reader = WADReader()

wad_with_features = reader.extract("../dataset/scraped/doom/01fava/FAVA.WAD")
wad = wad_with_features["wad"]
levels = wad_with_features["levels"]
features = levels[0]["features"]
maps = levels[0]["maps"]

# print(wad.keys())

for level in wad_with_features['wad'].levels:
#     print(level['lumps']['BLOCKMAP'])
    print(set(sector['floor_flat'] for sector in level['lumps']['SECTORS']))
#     # txrmap = np.zeros(mapsize_px, dtype=np.uint8)
#     list_of_sectors = list()
#     sectors = list(set(sidedef['sector'] for sidedef in level['lumps']['SIDEDEFS']))
#     for s in sectors:
#         sidedef_index = 0
#         list_of_sidedef = []
#         list_of_linedef = []
#         for sidedef in level['lumps']['SIDEDEFS']:
#             if sidedef['sector']==s:
#                 list_of_sidedef.append(sidedef_index)
#                 linedef_index = 0
#                 for linedef in level['lumps']['LINEDEFS']:
#                     if linedef['right_sidedef']==sidedef_index or linedef['left_sidedef']==sidedef_index:
#                         list_of_linedef.append(linedef_index)
#                     linedef_index += 1
#             sidedef_index += 1    
#         list_of_sectors.append({'SIDEDEFS': list_of_sidedef, 'LINEDEFS': list_of_linedef})
#         print("Sector", s, list_of_sectors[s])
# plt.imshow(maps["roommap"])
# plt.show()

