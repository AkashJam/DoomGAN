from WADEditor import WADReader
import os, json

dataset_path = '../dataset/scraped/doom/'
save_path = '../dataset/parsed/doom/'
json_path = dataset_path + 'doom.json'

unique_textures = list()
unique_flats = list()

# if os.path.isfile(save_path):
#     print('Collecting texture info...')
#     with open(save_path, 'r') as jsonfile:
#         texture_info = json.load(jsonfile)
#         unique_textures = list(texture_info['textures'].keys())
#         unique_flats = list(texture_info['flats'].keys())

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

wads = list()
for id in scraped_ids:
    wad_path = dataset_path + id + "/"
    # Listing all files in directories that have wad_id as their name
    for file in os.listdir(wad_path):
        if file.endswith(".WAD") or file.endswith(".wad"):
            try:
                wad = reader.read(wad_path + file)
                if len(wad['wad'].levels) == 1 and not wad['wad']['exception']:
                    # Searching for unique textures in SIDEDEFS lump
                    for i, dic in enumerate(wad['wad']['directory']):
                        if dic['name'] == 'SIDEDEFS':
                            for position in texture_position:
                                sidedef_tex = list(set(sidedef[position] for sidedef in wad['wad']['lumps'][i]))
                                for texture in sidedef_tex:
                                    if texture not in unique_textures:
                                        unique_textures.append(texture)
                                        # wads.append({'id':id, 'wad':file})
                                        print('Adding texture:', texture, 'from:',file)

                        if dic['name'] == 'SECTORS':
                            for position in flat_position:
                                for flat in list(set(sector[position] for sector in wad['wad']['lumps'][i])):
                                    if flat not in unique_flats:
                                        unique_flats.append(flat)
                                        # wads.append({'id':id, 'wad':file})
                                        print('Adding flat:', flat, 'from:',file)
            except:
                print('cannot parse level from WAD:',file)

unique_textures.sort(key=lambda v: v.upper())
unique_flats.sort(key=lambda v: v.upper())
graphic_meta = {'textures':len(unique_textures),'flats':len(unique_flats)}

texture_dict = {texture: i+1 for i,texture in enumerate(unique_textures)}
flat_dict = {flat: i+1 for i,flat in enumerate(unique_flats)}
textdict = {'textures': texture_dict, 'flats': flat_dict, 'meta':graphic_meta}


file = save_path + 'graphics.json'

with open(file, 'w') as jsonfile:
    json.dump(textdict, jsonfile)
