# Parser
Handles the translation of DOOM WADs into a set of feature maps as well as the conversion of the 

reader = WADReader()
wad_with_features = reader.extract('../dataset/scraped/doom/grendel1/GRENDEL1.WAD')
maps = wad_with_features["levels"][0]["maps"]
plt.subplots(2,4,tight_layout=True)
plt.subplot(2,4,1)
plt.imshow(maps["floormap"], cmap='gray')
plt.axis('off')
plt.subplot(2,4,2)
plt.imshow(maps["heightmap"], cmap='gray')
plt.axis('off')
plt.subplot(2,4,3)
plt.imshow(maps["wallmap"], cmap='gray')
plt.axis('off')
plt.subplot(2,4,4)
plt.imshow(maps["thingsmap"]["monsters"]+(maps["thingsmap"]["monsters"]>0).astype(int)*128, cmap='gray')
plt.axis('off')
plt.subplot(2,4,5)
plt.imshow(maps["thingsmap"]["ammunitions"]+(maps["thingsmap"]["ammunitions"]>0).astype(int)*128, cmap='gray')
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(maps["thingsmap"]["powerups"]+(maps["thingsmap"]["powerups"]>0).astype(int)*128, cmap='gray')
plt.axis('off')
plt.subplot(2,4,7)
plt.imshow(maps["thingsmap"]["artifacts"]+(maps["thingsmap"]["artifacts"]>0).astype(int)*128, cmap='gray')
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(maps["thingsmap"]["weapons"]+(maps["thingsmap"]["weapons"]>0).astype(int)*128, cmap='gray')
plt.axis('off')
plt.show()

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