import wget

url = input('Please enter image URL (string):')

file_name = wget.download(url)

print('Image Successfully Downloaded: ', file_name)