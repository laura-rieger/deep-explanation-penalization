from isic_api import ISICApi
import os
import json
import csv
with open('config.json') as json_file:
    data = json.load(json_file)

api = ISICApi(username=data["user"], password=data["pw"])
data_path = data["data_folder"]
num_imgs = data["num_imgs"]
if not os.path.exists(data_path):
    os.makedirs(data_path)
imageList = api.getJson('image?limit=' + str(num_imgs) +'&offset=0&sort=name')

#%%            
print('Fetching metadata for %s images' % len(imageList))
imageDetails = []
for image in imageList:
  
    # Fetch the full image details
    imageDetail = api.getJson('image/%s' % image['_id'])
    imageDetails.append(imageDetail)

# Determine the union of all image metadata fields
metadataFields = set(
        field
        for imageDetail in imageDetails
        for field in imageDetail['meta']['clinical'].keys()
    )


metadataFields = ['isic_id'] + sorted(metadataFields)
outputFileName = "meta"
#%%
outputFilePath = os.path.join(data_path, outputFileName)
# Write the metadata to a CSV
print('Writing metadata to CSV: %s' % outputFileName+'.csv')
with open(outputFilePath+'.csv', 'w') as outputStream:
    csvWriter = csv.DictWriter(outputStream, metadataFields)
    csvWriter.writeheader()
    for imageDetail in imageDetails:
        rowDict = imageDetail['meta']['clinical'].copy()
        rowDict['isic_id'] = imageDetail['name']
        csvWriter.writerow(rowDict)