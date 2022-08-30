# import urllib.request # works with ftp servers
import requests
from bs4 import BeautifulSoup

def openPage(url):
    session = requests.Session()
    page = session.get(url).text
    return BeautifulSoup(page, "lxml")


repo_link = "https://www.doomworld.com/idgames/levels/doom/"
# list of subcategories to be avoided
excluded_list = ["deathmatch","Ports","megawads"]
soup = openPage(repo_link)
# Fetches subcategory url in doomworld and proceeds for just one level down the tree. 
links = soup.find_all('a')
for l in links:
    if l.has_attr('href') and "idgames/levels/doom" in l['href']:
        url = l['href']
        
        if any(x in url for x in excluded_list):
            continue

        id = l['href'].split('/')[-2]
        # print(id,url)

        suburl = repo_link + id + "/"
        subsoup = openPage(suburl)
        print(suburl)
        sublinks = subsoup.find_all('a')
        for sl in sublinks:
            if sl.has_attr('href') and sl['href'].startswith("levels/doom"):
                uri = sl['href']
                name = sl.contents[0]
                id = sl['href'].split('/')[-1]
                print(id,uri)

    

# urllib.request.urlretrieve('ftp://ftp.fu-berlin.de/pc/games/idgames/levels/doom/0-9/0.zip', './scraped/doom/0.zip')