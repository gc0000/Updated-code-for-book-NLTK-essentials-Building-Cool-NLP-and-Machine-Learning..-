# code changed

import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import Selector
from scrapy.item import NewsItem
from scrapy.item import Item, Field

# a basic spider
class NewsSpider(scrapy.Spider):
    name = "news"
    allowed_domains = ["sina.com"]
    start_urls = [
    'https://news.sina.com.cn/'
     ]
    rules = (
        # Extract links matching cnn.com
        Rule(SgmlLinkExtractor(allow=('cnn.com',), deny = ('http: //edition.cnn.com / ', ))),
        # Extract links matching 'news.google.com'
        Rule(SgmlLinkExtractor(allow=('news.google.com',)),
        callback = 'parse_news_item'),
    )
    def parse_news_item(self, response):
        sel = Selector(response)
        item = NewsItem()
        item['title'] = sel.xpath('//title/text()').extract()
        item[topic] = sel.xpath('/div[@class="topic"]').extract()
        item['desc'] = sel.xpath('//td//text()').extract()
        return item
        # sites = sel.xpath('//ul/li')
        # for site in sites:
        #     title = site.xpath('a/text()').extract()
        #     link = site.xpath('a/@href').extract()
        #     desc = site.xpath('text()').extract()
        #     print(title, link, desc)
class NewsItem(Item):
    title=Field()
    link=Field()
    desc=Field()

# local parsing
from scrapy.contrib.spiders import SitemapSpider
class MySpider(SitemapSpider):
    sitemap_URLss = ['http://www.example.com/sitemap.xml']
    sitemap_rules = [('/electronics/', 'parse_electronics'), ('apparel / ', 'parse_apparel')
        ,]

    def parse_electronics(self, response):
        # you need to create an item for electronics,
        return
    def parse_apparel(selfself, response):
        # you need to create an item for apparel
        return

# automatic login
class LoginSpider(BaseSpider):
    name = 'example.com'
    start_URLss = ['http://www.example.com/users/login.php']
    def parse(self, response):
        return [FormRequest.from_response(response,
                                          formdata={'username': 'john', 'password': 'secret'},
                                          callback=self.after_login)]

    def after_login(self, response):
        # check login succeed before going on
        if "authentication failed" in response.body:
            self.log("Login failed", level=log.ERROR)
        return

# derive the age from DOB
from scrapy.item import Item
import datetime
class AgePipeline(object):
    def process_item(self, item, spider):
        if item['DOB']:
            item['Age'] = (datetime.datetime.
                           strptime(item['DOB'], '%d-%m-%y').date() - datetime.datetime.
                           strptime('currentdate', ' % d - % m -% y').date()).days/365
        return item

# remove duplicates
from scrapy import signals
class DuplicatesPipeline(object):
    def __init__(selfself):
        self.ids_seen = set()
    def process_item(selfself, item, spider):
        if item['id'] in self.ids_seen:
            raise DropItem("Duplicate item found: %s" % item)
        else:
            self.ids_seen.add(item['id'])
            return item

# write the item in the JSON file using JsonWriterPipeline.py pipeline
import json
class JsonWriterPipeline(object):
    def __init__(selfself):
        self.file = open('items.txt', 'wb')
    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item






