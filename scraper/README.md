# Scraper
DOOM game files are stored as WADs and are acquired through scrapping the 'doomworld' website using beautifulsoup

The code below demonstrates how to download the DOOM WADs from the repository using available functions
```
eg = 'https://www.doomworld.com/idgames/levels/doom/a-c/acastle2'
download_wad(eg,save_path)
```

downloaded files are compressed and can be extracted by providing the path and name
```
with zipfile.ZipFile('../dataset/scraped/doom/acastle2/' + 'acastle2.zip', 'r') as zip_ref:
            zip_ref.extractall('../dataset/scraped/doom/acastle2/')
```

The file mentioned in this folder automatically scraps available DOOM levels and keeps track of the downloaded files through the DOOM json at the location where it is saved. To begin scrapping the levels run:
```
python scrapper.py
```