from isic_api import ISICApi
#Insert Username and Password Below
# The ISIC dataset is freely available for research use. You have to create a user on isic-archive.com
api = ISICApi(username="<INSERT USERNAME>", password="<INSERT PASSWORD>")

imageList = api.getJson('image?limit=100&offset=0&sort=name')

import urllib
import os
savePath = 'ISICArchive/'

if not os.path.exists(savePath):
    os.makedirs(savePath)
start_offset = 14000
for i in range(300):
    
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
            
            
            
#print('Fetching metadata for %s images' % len(imageList))
#imageDetails = []
#for image in imageList:
#    print(' ', image['name'])
#    # Fetch the full image details
#    imageDetail = api.getJson('image/%s' % image['_id'])
#    imageDetails.append(imageDetail)
#
## Determine the union of all image metadata fields
#metadataFields = set(
#    field
#    for imageDetail in imageDetails
#    for field in imageDetail['meta']['clinical'].viewkeys()
#)
#metadataFields = ['isic_id'] + sorted(metadataFields)
#
## Write the metadata to a CSV
#print('Writing metadata to CSV: %s' % outputFileName+'.csv')
#with open(outputFilePath+'.csv', 'wb') as outputStream:
#    csvWriter = csv.DictWriter(outputStream, metadataFields)
#    csvWriter.writeheader()
#    for imageDetail in imageDetails:
#        rowDict = imageDetail['meta']['clinical'].copy()
#        rowDict['isic_id'] = imageDetail['name']
#        csvWriter.writerow(rowDict)