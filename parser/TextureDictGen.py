from WADEditor import WADReader
import os, json

dataset_path = '../dataset/scraped/doom/'
save_path = './Dictionaries/TextureTypes.json'
json_path = dataset_path + 'doom.json'

unique_textures = list()
unique_flats = list()

if os.path.isfile(save_path):
    print('Collecting texture info...')
    with open(save_path, 'r') as jsonfile:
        texture_info = json.load(jsonfile)
        unique_textures = list(texture_info['textures'].keys())
        unique_flats = list(texture_info['flats'].keys())

if os.path.isfile(json_path):
    print('Collecting scraped info...')
    with open(json_path, 'r') as jsonfile:
        scraped_info = json.load(jsonfile)
        print('Loaded {} records.'.format(len(scraped_info)))
        if len(scraped_info) != 0:
            scraped_ids = [info['id'] for info in scraped_info if 'id' in info]

reader = WADReader()
texture_position = ['upper_texture','middle_texture','lower_texture']
flat_position = ['floor_flat', 'ceiling_flat']

# wads = list()
for id in scraped_ids:
    wad_path = dataset_path + id + "/"
    # Listing all files in directories that have wad_id as their name
    for file in os.listdir(wad_path):
        if file.endswith(".WAD") or file.endswith(".wad"):
            wad = reader.read(wad_path + file)

            # Searching for unique textures in SIDEDEFS lump
            for i, dic in enumerate(wad['wad']['directory']):
                if dic['name'] == 'SIDEDEFS':
                    for position in texture_position:
                        for texture in list(set(sidedef[position] for sidedef in wad['wad']['lumps'][i])):
                            if texture not in unique_textures:
                                unique_textures.append(texture)
                                # wads.append({'id':id, 'wad':file})
                                print('Adding texture', texture)

                if dic['name'] == 'SECTORS':
                    for position in flat_position:
                        for flat in list(set(sector[position] for sector in wad['wad']['lumps'][i])):
                            if flat not in unique_flats:
                                unique_flats.append(flat)
                                # wads.append({'id':id, 'wad':file})
                                print('Adding flat', flat)

unique_textures.sort(key=lambda v: v.upper())
unique_flats.sort(key=lambda v: v.upper())
texture_dict = {texture: i for i,texture in enumerate(unique_textures)}
flat_dict = {flat: i for i,flat in enumerate(unique_flats)}
textdict = {'textures': texture_dict, 'flats': flat_dict}

# # print(textdict)
with open(save_path, 'w') as jsonfile:
    json.dump(textdict, jsonfile)

    
# [{'filepos': 12, 'size': 0, 'name': 'E1M1'}, {'filepos': 12, 'size': 8620, 'name': 'THINGS'}, {'filepos': 8632, 'size': 18312, 'name': 'LINEDEFS'}, 
# {'filepos': 26944, 'size': 52950, 'name': 'SIDEDEFS'}, {'filepos': 79894, 'size': 5480, 'name': 'VERTEXES'}, {'filepos': 85374, 'size': 24024, 'name': 'SEGS'}, 
# {'filepos': 109398, 'size': 2120, 'name': 'SSECTORS'}, {'filepos': 111518, 'size': 14812, 'name': 'NODES'}, {'filepos': 126330, 'size': 5590, 'name': 'SECTORS'}, 
# {'filepos': 131920, 'size': 5779, 'name': 'REJECT'}, {'filepos': 137699, 'size': 36396, 'name': 'BLOCKMAP'}]