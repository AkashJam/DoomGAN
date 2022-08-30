import urllib.request
import requests
from bs4 import BeautifulSoup

session = requests.Session()
def open_page(url):
    page = session.get(url).text
    return BeautifulSoup(page, "lxml")

repo_link = 'https://www.doomworld.com/idgames/levels/doom/'
# list of subcategories to be avoided
excluded_list = ['deathmatch','Ports','megawads']
level_page = open_page(repo_link)
# Fetching subcategory url in doomworld 
level_page_links = level_page.find_all('a')
for l in level_page_links:
    # Finding urls that contain this path
    if l.has_attr('href') and 'idgames/levels/doom' in l['href']:
        cate_path = l['href']
        
        # Skipping excluded categories
        if any(x in cate_path for x in excluded_list):
            continue

        cate_id = l['href'].split('/')[-2]

        cate_url = repo_link + cate_id + "/"
        # print(cate_url)
        # Fetching links of levels in subcategory page
        cate_page = open_page(cate_url)
        cate_page_links = cate_page.find_all('a')
        for c in cate_page_links:
            if c.has_attr('href') and c['href'].startswith('levels/doom'):
                map_path = c['href']
                map_id = c['href'].split('/')[-1]

                # Fetching download links of WAD files
                found_files = []
                map_url = cate_url + map_id
                print ('Opening map page: ' + map_url)
                map_page = open_page(map_url)
                # Finds the table containing the WAD download links
                mainDiv = map_page.findAll('table', {'class': 'download'})[0]
                file_page_links = [entry.find_all('a') for entry in mainDiv.find_all('ul', {'class':'square'})][0]
                filename = file_page_links[0]['href'].split('/')[-1]
                for file in file_page_links:
                    print(file['href'])
                    link = file['href']
                    if link[0] =='f':
                        try:
                            # for ftp servers
                            urllib.request.urlretrieve(file['href'], '../dataset/scraped/doom/' + filename)
                        except Exception as e:
                            print(e)
                    else:
                        r = session.get(link)
                        if r.status_code != 200:
                            print ('Failed to download the file: ' + link)
                        # filename = r.url.split('/')[-1]
                        with open('../dataset/scraped/doom/' + filename, 'wb') as file:
                            file.write(r.content)
                    break
                break
        break
                    # print("LINK",file['href'])
                    # break
                # found_files = found_files + [fpl['href'] for fpl in file_page_links]
                # print("Found {} more files".format(str(len(found_files))))
                # print(found_files)

# for ftp servers
# urllib.request.urlretrieve('ftp://ftp.fu-berlin.de/pc/games/idgames/levels/doom/0-9/0.zip', './scraped/doom/0.zip')

# for http servers
# url = "https://www.gamers.org/pub/idgames/levels/doom/0-9/0.zip"
# local_path = "./"
# session = requests.Session()
# r = session.get(url)
# if r.status_code != 200:
#     print ("Failed to download the file: " + url)
# filename = r.url.split('/')[-1]
# with open("./scraped/doom/"+filename, "wb") as file:
#     file.write(r.content)