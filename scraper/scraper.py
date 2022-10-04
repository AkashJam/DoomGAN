import urllib.request
import json
import os
import requests
from bs4 import BeautifulSoup
import re
import zipfile


def open_page(url):
    page = session.get(url).text
    return BeautifulSoup(page, "lxml")

def fetch_subcategories_links(cate_url,exclude):
    cate_page = open_page(cate_url)
    cate_links = cate_page.find_all('a')
    subcate_urls = []
    for c in cate_links:

        # Finding urls that contain this path
        if c.has_attr('href') and 'idgames/levels/doom' in c['href']:
            subcate_path = c['href']
            
            # Skipping excluded categories
            if any(x in subcate_path for x in exclude):
                continue

            subcate_id = c['href'].split('/')[-2]
            subcate_urls += [cate_url + subcate_id + "/"]

    return subcate_urls

def fetch_level_links(subcate_urls):
    level_urls=[]
    for subcate in subcate_urls:
        subcate_page = open_page(subcate)
        # Fetching links of individual levels in subcategory page
        subcate_page_link = subcate_page.find_all('a')
        for s in subcate_page_link:
            if s.has_attr('href') and s['href'].startswith('levels/doom'):
                level_id = s['href'].split('/')[-1]
                level_urls += [subcate + level_id]
    return level_urls

def fetch_level_info(level_url):
    level_page = open_page(level_url)
    info_list = level_page.findAll('td', {'class': re.compile('filelist_field')})
    stars = info_list[-1].findAll('img', {'src': re.compile('star')})
    ratings = 0
    for x in stars:
        if x['src'] == 'images/star.gif':
            ratings+=1
        elif x['src'] == 'images/halfstar.gif':
            ratings+=0.5
        elif 'empty' not in x['src'] :
            ratings+=0.25
    level_info = dict()
    level_info['id'] = level_url.split('/')[-1]
    level_info['name'] = info_list[0].contents[0][1:]
    level_info['url'] = level_url
    level_info['rating_value'] = ratings
    # print(info_list[-1].content)
    level_info['rating_count'] = int(re.findall(r'\d+', info_list[-1].contents[-1])[0]) if info_list[-1].content is not None else 0
    return level_info
    # scraped_info.append(level_info)
    # with open(json_path, 'w') as jsonfile:
    #    json.dump(scraped_info, jsonfile)



def download_wad(level_url, download_path):
    level_page = open_page(level_url)
    # Finds the table containing the WAD download links
    mainDiv = level_page.findAll('table', {'class': 'download'})[0]
    level_page_links = [entry.find_all('a') for entry in mainDiv.find_all('ul', {'class':'square'})][0]
    downloaded = False
    for file in level_page_links:
        link = file['href']
        filename = link.split('/')[-1]
        file_id = level_url.split('/')[-1]
        # print(filename,link,level_url,file_id)
        file_path = download_path + "/" + file_id + "/"
        if os.path.exists(file_path):
            if os.path.exists(file_path + filename):
                return False
        else:
            os.makedirs(file_path)
        # for ftp servers
        if link[0:3] =='ftp':
            try:
                urllib.request.urlretrieve(link, file_path + filename)
                downloaded = True
                break
            except Exception as e:
                print ('Failed to download the file: ', link)
        else:
            # for http servers
            r = session.get(link)
            if r.status_code != 200:
                print ('Failed to download the file: ', link)
                continue
            with open(file_path + filename, 'wb') as file:
                file.write(r.content)
                downloaded = True
                break
    if downloaded:
        # Extract ZIP files
        try:
            zip_ref = zipfile.ZipFile(file_path + filename, 'r')
            zip_ref.extractall(file_path)
        except Exception as e:
            print ('Failed to extract the file', filename)
    return downloaded

    # Need to return true if downloaded else false to see if the info needs to be saved into the doom json file
    



session = requests.Session()
archived_cate = 'https://www.doomworld.com/idgames/levels/doom/'
# List of subcategories to be avoided
excluded_list = ['deathmatch','Ports','megawads']
save_path = '../dataset/scraped/doom/'


# Create dataset directory
if os.path.exists(save_path):
    print('found location to store scraped files')
else:
    os.makedirs(save_path)

# Check if the json file is present and try to resume downloading if possible
scraped_info = list()
visited_links = list()
json_path = save_path + 'doom.json'
if os.path.isfile(json_path):
    print('Trying to resume download...')
    with open(json_path, 'r') as jsonfile:
        scraped_info = json.load(jsonfile)
        print('Loaded {} records.'.format(len(scraped_info)))
        if len(scraped_info) != 0:
            visited_links = [info['url'] for info in scraped_info if 'url' in info]

# print(scraped_info,visited_links)

# Fetching subcategory url in doomworld 
sub_links = fetch_subcategories_links(archived_cate,excluded_list)
level_links = fetch_level_links(sub_links)
for level_link in level_links:
    if level_link in visited_links:
        print('skipping ',level_link)
        continue
    print('downloading level from ',level_link)
    status = download_wad(level_link,save_path)
    if status:
        info = fetch_level_info(level_link)
        print('downloading level info from ',level_link)
        scraped_info.append(info)
        with open(json_path, 'w') as jsonfile:
            json.dump(scraped_info, jsonfile)

# with zipfile.ZipFile('../dataset/scraped/doom/crescent/' + 'crescent.zip', 'r') as zip_ref:
#             zip_ref.extractall('../dataset/scraped/doom/crescent/')

# eg = 'https://www.doomworld.com/idgames/levels/doom/a-c/alleve'

# info = fetch_level_info(eg)
# print(info)
# download_wad(eg,save_path)
# https://www.doomworld.com/idgames/levels/doom/a-c/acastle2
    



