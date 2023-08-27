from utils.get_med_class import get_medicine_class
from utils.get_med_query import get_query

IMAGE_URL = 'https://cdn11.bigcommerce.com/s-4fff2/images/stencil/640w/products/88068/876840/Adol500mgCaplets96s__71446.1692720817.jpg?c=2'


print(get_medicine_class(IMAGE_URL))
print(get_query("Side Effects","adol"))