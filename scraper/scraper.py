import urllib.request
import requests
from bs4 import BeautifulSoup


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

def fetch_level_info(level_urls):
    for level in level_urls:
        level_page = open_page(level)
        info_table = level_page.findAll('table', {'class': 'filelist'})
        print(info_table)


def download_wad(level_urls, download_path):
    for level in level_urls:
        level_page = open_page(level)
        # Finds the table containing the WAD download links
        mainDiv = level_page.findAll('table', {'class': 'download'})[0]
        level_page_links = [entry.find_all('a') for entry in mainDiv.find_all('ul', {'class':'square'})][0]
        for file in level_page_links:
            filename = file['href'].split('/')[-1]
            link = file['href']
            # for ftp servers
            if link[0] =='f':
                try:
                    urllib.request.urlretrieve(file['href'], download_path + filename)
                    break
                except Exception as e:
                    print(e)
            else:
                # for http servers
                r = session.get(link)
                if r.status_code != 200:
                    print ('Failed to download the file: ' + link)
                    continue
                with open(download_path + filename, 'wb') as file:
                    file.write(r.content)
                    break



session = requests.Session()
archived_cate = 'https://www.doomworld.com/idgames/levels/doom/'
# list of subcategories to be avoided
excluded_list = ['deathmatch','Ports','megawads']
save_path = "../dataset/scraped/doom/"
# Fetching subcategory url in doomworld 
# sub_links = fetch_subcategories_links(archived_cate,excluded_list)
# level_links = fetch_level_links(sub_links)
eg = ['https://www.doomworld.com/idgames/levels/doom/a-c/bak2hell']
info = fetch_level_info(eg)




# if os.path.exists(root_path):
#             print("continuing scrapping")
#         else:
#             os.makedirs(root_path)

# # Check if the json file is present and try to resume downloading if possible
#         json_path = root_path + game_name + '.json'
#         if os.path.isfile(json_path):
#             print("Trying to resume download...")
#             with open(json_path, 'r') as jsonfile:
#                 self.file_info = json.load(jsonfile)
#                 print("Loaded {} records.".format(len(self.file_info)))






# levels_page = open_page(archive_cate)
# levels_page_links = levels_page.find_all('a')
# for l in levels_page_links:
#     # Finding urls that contain this path
#     if l.has_attr('href') and 'idgames/levels/doom' in l['href']:
#         cate_path = l['href']
        
#         # Skipping excluded categories
#         if any(x in cate_path for x in excluded_list):
#             continue

#         cate_id = l['href'].split('/')[-2]

#         cate_url = repo_link + cate_id + "/"
#         # print(cate_url)
#         cate_page = open_page(cate_url)
#         # Fetching links of individual levels in subcategory page
#         cate_page_links = cate_page.find_all('a')
#         for c in cate_page_links:
#             if c.has_attr('href') and c['href'].startswith('levels/doom'):
#                 map_path = c['href']
#                 map_id = c['href'].split('/')[-1]
#                 map_url = cate_url + map_id
#                 print ('Opening level page: ' + map_url)
#                 map_page = open_page(map_url)
#                 # Finds the table containing the WAD download links
#                 mainDiv = map_page.findAll('table', {'class': 'download'})[0]
#                 file_page_links = [entry.find_all('a') for entry in mainDiv.find_all('ul', {'class':'square'})][0]
#                 filename = file_page_links[0]['href'].split('/')[-1]
#                 # for file in file_page_links:
#                 #     link = file['href']
#                 #     if link[0] =='h':
#                 #         # print(link)
#                 #         if link[0] =='f':
#                 #             try:
#                 #                 # for ftp servers
#                 #                 urllib.request.urlretrieve(file['href'], save_path + filename)
#                 #                 break
#                 #             except Exception as e:
#                 #                 print(e)
#                 #         else:
#                 #             # for http servers
#                 #             r = session.get(link)
#                 #             if r.status_code != 200:
#                 #                 print ('Failed to download the file: ' + link)
#                 #                 continue
#                 #             with open(save_path + filename, 'wb') as file:
#                 #                 file.write(r.content)
#         #                     break
#         #         break
#         # break