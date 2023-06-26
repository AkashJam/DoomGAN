import urllib.request, json, os, requests, re, zipfile
from bs4 import BeautifulSoup


class Scraper:
    def __init__(self):
        super().__init__()
        self.session = requests.Session() # required to download files from http servers
        self.archived_cate = 'https://www.doomworld.com/idgames/levels/doom/'
        self.excluded_list = ['deathmatch','Ports','megawads'] # List of subcategories to be avoided
        self.save_path = 'dataset/scraped/doom/'


    def open_page(self, url):
        page = self.session.get(url).text
        return BeautifulSoup(page, "lxml")


    def fetch_subcategories_links(self, cate_url):
        cate_page = self.open_page(cate_url)
        cate_links = cate_page.find_all('a')
        subcate_urls = []
        for c in cate_links:
            # Finding urls that contain this path
            if c.has_attr('href') and 'idgames/levels/doom' in c['href']:
                subcate_path = c['href']
                # Skipping excluded categories
                if any(x in subcate_path for x in self.excluded_list):
                    continue
                subcate_id = c['href'].split('/')[-2]
                subcate_urls += [cate_url + subcate_id + "/"]
        return subcate_urls


    def fetch_level_links(self, subcate_urls):
        level_urls=[]
        for subcate in subcate_urls:
            subcate_page = self.open_page(subcate)
            # Fetching links of individual levels in subcategory page
            subcate_page_link = subcate_page.find_all('a')
            for s in subcate_page_link:
                if s.has_attr('href') and s['href'].startswith('levels/doom'):
                    level_id = s['href'].split('/')[-1]
                    level_urls += [subcate + level_id]
        return level_urls


    def fetch_level_info(self, level_url):
        level_page = self.open_page(level_url)
        # searching for the level info
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
        level_info['rating_count'] = int(re.findall(r'\d+', info_list[-1].contents[-1])[0]) if info_list[-1].content is not None else 0
        return level_info



    def download_wad(self, level_url):
        level_page = self.open_page(level_url)
        # Finds the table containing the WAD download links
        mainDiv = level_page.findAll('table', {'class': 'download'})[0]
        level_page_links = [entry.find_all('a') for entry in mainDiv.find_all('ul', {'class':'square'})][0]
        downloaded = False
        for file in level_page_links:
            link = file['href']
            filename = link.split('/')[-1]
            file_id = level_url.split('/')[-1]
            # print(filename,link,level_url,file_id)
            file_path = self.save_path + "/" + file_id + "/"
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
                r = self.session.get(link)
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
        # Need to return true if downloaded else false to see if the info needs to be saved into the doom json file
        return downloaded


    def scrap_levels(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        scraped_info = list()
        visited_links = list()
        json_path = self.save_path + 'doom.json'
        if os.path.isfile(json_path):
            print('Trying to resume download...')
            with open(json_path, 'r') as jsonfile:
                scraped_info = json.load(jsonfile)
                print('Loaded {} records.'.format(len(scraped_info)))
                if len(scraped_info) != 0:
                    # Check if the json file is present and try to resume downloading if possible
                    visited_links = [info['url'] for info in scraped_info if 'url' in info]

        # Fetching subcategory url in doomworld 
        sub_links = self.fetch_subcategories_links(self.archived_cate)
        level_links = self.fetch_level_links(sub_links)

        for level_link in level_links:
            if level_link in visited_links:
                print('skipping ',level_link)
                continue
            print('downloading level from ',level_link)
            status = self.download_wad(level_link)
            if status:
                info = self.fetch_level_info(level_link)
                print('downloading level info from ',level_link)
                scraped_info.append(info)
                with open(json_path, 'w') as jsonfile:
                    json.dump(scraped_info, jsonfile)