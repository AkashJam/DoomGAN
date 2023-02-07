# Parser
DOOM WADs consists of 


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