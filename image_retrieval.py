from bs4 import BeautifulSoup
import requests
import boto3
from numpy.random import choice

bk = 'donut-corners-images'

def get_image(name = 'random', src = 'local'):
    # if src == 'local':
    #     return None
    # if src == 's3':
    #     return None
    # if src == 'google':
    #     return None
    if src == 'ebay':
        return get_img_ebay(name)
    
    return f'source not found: {src}'


def get_img_ebay(term = 'furniture', n = 1):
    return download(get_urls_ebay(term, n))


def get_urls_ebay(term = 'furniture', n = 1):
    urls = []
    page = 1
    while len(urls) < n:
        r = requests.get(f"https://www.ebay.com/sch/i.html?_from=R40&_nkw={term}&_sacat=0&_pgn={page}")
        soup = BeautifulSoup(r.content, 'html.parser')
        batch = [item.find('img').get('src') for item in soup.select('.s-item__image')]
        batch = ['2000'.join(url.rsplit('225', 1)) for url in batch]
        urls.extend(batch)
        page += 1
    
    return choice(urls, n, replace=False)


def download(urls):
    return [requests.get(url).content for url in urls]


def write_disk(images, filenames, folder):
    for img, fn in zip(images, filenames):
        with open(folder + fn, 'wb') as out:
            out.write(img)


def write_s3(images, filenames):
    s3 = boto3.client('s3')
    for img, fn in zip(images, filenames):
        s3.put_object(Bucket='bk', Body=img, Key=fn)