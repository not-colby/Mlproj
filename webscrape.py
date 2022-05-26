from concurrent.futures import ThreadPoolExecutor
from re import I
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import io
from PIL import Image
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import threading
import concurrent.futures.thread
from concurrent.futures import ThreadPoolExecutor
# PATH = os.getcwd()
#PATH = PATH + "/chrome_driver/chromedriver.exe"
#print("/chrome_driver/chromedriver.exe")
#wd = webdriver.Chrome(PATH)
wd = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

def get_images_from_google(wd, delay, max_images):
	def scroll_down(wd):
		wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
		time.sleep(delay)
	# Agate
	#url ="https://www.google.com/search?q=agate&client=ubuntu&hs=ZOu&channel=fs&source=lnms&tbm=isch&sa=X&ved=2ahUKEwig_5mMgP33AhVeoI4IHRmAC7gQ_AUoAXoECAIQAw&biw=1920&bih=995&dpr=1"
	# Quartz
	# url = "https://www.google.com/search?q=quartz+or+clear+quartz++or+crystal+quatz&tbm=isch&ved=2ahUKEwjBjJn_hf33AhUGXs0KHfJ_AY4Q2-cCegQIABAA&oq=quartz+or+clear+quartz++or+crystal+quatz&gs_lcp=CgNpbWcQA1DCHVjMN2C1OWgBcAB4AIABiwGIAZoJkgEEMTYuMZgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=cWKPYoGQNoa8tQby_4XwCA&bih=718&biw=1727&client=ubuntu&hs=ZOu&hl=en-US"
	# Obsidian
	#url ="https://www.google.com/search?q=obsidian+rock+or+obsidian+or+obsidian+mineral+-minecraft+-movie+&tbm=isch&ved=2ahUKEwjNpebvjf33AhVRCJ0JHQrrDGYQ2-cCegQIABAA&oq=obsidian+rock+or+obsidian+or+obsidian+mineral+-minecraft+-movie+&gs_lcp=CgNpbWcQA1CUBVj4jwFgqZcBaAVwAHgAgAFJiAGdFZIBAjQ0mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=tWqPYo33BNGQ9PwPitazsAY&bih=704&biw=1386&client=ubuntu&hs=gLv"
	# Granite
	#url ="https://www.google.com/search?q=granite+or+granite+ore+or+granite+mineral+-marble+-tile+-table+-counter+-truck+-runescape+-oregon&tbm=isch&ved=2ahUKEwjQ99utkP33AhULBs0KHTNeAtAQ2-cCegQIABAA&oq=granite+or+granite+ore+or+granite+mineral+-marble+-tile+-table+-counter+-truck+-runescape+-oregon&gs_lcp=CgNpbWcQA1AAWPqoAWCtqwFoBnAAeACAAYIBiAHvEJIBBDI5LjOYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=T22PYpCUMYuMtAazvImADQ&bih=704&biw=866&client=ubuntu&hl=en-US"
	
	wd.get(url)

	image_urls = set()
	skips = 0

	while len(image_urls) + skips < max_images:
		scroll_down(wd)

		thumbnails = wd.find_elements(By.CLASS_NAME, "Q4LuWd")

		for img in thumbnails[len(image_urls) + skips:max_images]:
			try:
				img.click()
				time.sleep(delay)
			except:
				continue

			images = wd.find_elements(By.CLASS_NAME, "n3VNCb")
			for image in images:
				if image.get_attribute('src') in image_urls:
					max_images += 1
					skips += 1
					break

				if image.get_attribute('src') and 'http' in image.get_attribute('src'):
					image_urls.add(image.get_attribute('src'))
					print(f"Found {len(image_urls)}")

	return image_urls


def download_image(download_path, url, file_name):
	try:
		image_content = requests.get(url).content
		image_file = io.BytesIO(image_content)
		image = Image.open(image_file)
		file_path = download_path + file_name

		with open(file_path, "wb") as f:
			image.save(f, "JPEG")

		print("Success")
	except Exception as e:
		print('FAILED -', e)

urls = get_images_from_google(wd, 1, 450)
pool = ThreadPoolExecutor(20)
for i, url in enumerate(urls):
	pool.submit(download_image, "temp_imgs/", url, str(i) + ".jpg")
	#download_image("temp_imgs/", url, str(i) + ".jpg")
	

wd.quit()
