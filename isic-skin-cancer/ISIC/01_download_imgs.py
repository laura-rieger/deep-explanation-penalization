from isic_api import ISICApi
import os
import json
with open('config.json') as json_file:
    data = json.load(json_file)

api = ISICApi(username=data["user"], password=data["pw"])
data_path = data["data_folder"]
num_imgs = data["num_imgs"]
#%%
savePath = os.path.join(data_path, 'raw')

if not os.path.exists(savePath):
    os.makedirs(savePath)
start_offset = 0
#%%

for i in range(int(num_imgs/50)+1):
    
    imageList = api.getJson('image?limit=50&offset=' + str(start_offset) + '&sort=name')
    
    print('Downloading %s images' % len(imageList))
    
    for image in imageList:
        print(image['_id'])
        imageFileResp = api.get('image/%s/download' % image['_id'])
        imageFileResp.raise_for_status()
        imageFileOutputPath = os.path.join(savePath, '%s.jpg' % image['name'])
        with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
            for chunk in imageFileResp:
                imageFileOutputStream.write(chunk)
    start_offset +=50
            
            
#%%

