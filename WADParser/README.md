# Parser
Handles the translation of DOOM WADs to a dictionary structure as well as the conversion of them into a set of feature maps.
```
reader = WADReader()
wad_with_features = reader.extract('../dataset/scraped/doom/grendel1/GRENDEL1.WAD')
maps = wad_with_features["levels"][0]["maps"]
feature_keys = ['floormap', 'wallmap', 'heightmap', 'triggermap', 'roommap', 'thingsmap', 'texturemap']
categories = ['essentials','start', 'other', 'keys','obstacles',  'monsters', 'ammunitions', 'weapons', 'powerups', 'artifacts']
textures = ['floortexturemap', 'ceilingtexturemap', 'rightwalltexturemap','leftwalltexturemap']
plt.subplots(2,4,tight_layout=True)
for i,key in enumerate(list(maps.keys())):
    if key == 'thingsmap':
        for cate in categories
            plt.subplot(2,4,i)
            plt.imshow(maps[key][cate]+(maps[key][cate]>0).astype(int)*128, cmap='gray')
            plt.axis('off')
    elif key != 'textures:
        plt.subplot(2,4,i)
        plt.imshow(maps[key], cmap='gray')
        plt.axis('off')
plt.show()
```

ZenNode is used in the final process to generate the remaining aspects of the DOOM WAD to be playable by the DOOM engine such as predefined computation like the BLOCKMAP and REJECT lumps
```
from WADEditor import WADWriter 
# Let's create a new WAD
writer = WADWriter()
# Declare a level
writer.add_level('MAP01')
# Create a big sector, by specifying its vertices (in clockwise order)
writer.add_sector([(1000,1000),(1000,-1000), (-1000,-1000), (-1000,1000) ])
# set the starting position for the player 1
writer.set_start(0,0)
# Let's add a Cacodemon to make things more interesting
writer.add_thing(x=500,y=500,thing_type=3005, options=7) 
# Save the wad file. "bsp" command should work in your shell for this.
wad_mine = writer.save('test.wad')
```