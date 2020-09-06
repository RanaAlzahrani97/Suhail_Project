#%%
#import all libraries
import pandas as pd
import numpy as np
import face_recognition
import cv2
import os
import csv
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from two_lists_similarity import Calculate_Similarity as cs


#%%
#Reading all files 
data = pd.read_csv('C:\\Users\\RanaA\\PycharmProjects\\pythonProject\\listing.csv', encoding = 'latin-1')
desc = pd.read_csv('C:\\Users\\RanaA\\PycharmProjects\\pythonProject\\descrption.csv', encoding = 'latin-1')
users = pd.read_csv('C:\\Users\\RanaA\\PycharmProjects\\pythonProject\\Users.csv', encoding = 'latin-1')
data2 = pd.read_csv('C:\\Users\\RanaA\\PycharmProjects\\pythonProject\\Users.csv', index_col="Name")
data3 = pd.read_csv('C:\\Users\\RanaA\\PycharmProjects\\pythonProject\\descrption.csv', index_col="ID")
data4 = pd.read_csv('C:\\Users\\RanaA\\PycharmProjects\\pythonProject\\descrption.csv', index_col="Name")
data5 = pd.read_csv('C:\\Users\\RanaA\\PycharmProjects\\pythonProject\\Users.csv', index_col="Name")
data6 = pd.read_csv('C:\\Users\\RanaA\\PycharmProjects\\pythonProject\\Object.csv', index_col="ID")
object = pd.read_csv('C:\\Users\\RanaA\\PycharmProjects\\pythonProject\\Object.csv')

#%%
# face recognition using live camera 
j=0
userName=''
Encodings=[]
Names=[]
dispW=640
dispH=480
flip=2
 
with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)
font=cv2.FONT_HERSHEY_SIMPLEX
#for Rasperrypi Camera v2 
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance  contrast=1.5 brightness=-.3 saturation=1.2 ! appsink  '
cam= cv2.VideoCapture(camSet)
 
while True:
 
    _,frame=cam.read()
    frameSmall=cv2.resize(frame,(0,0),fx=.25,fy=.25)
    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions=face_recognition.face_locations(frameRGB,model='cnn')
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions)
    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
        name='Unkown Person'
        matches=face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            first_match_index=matches.index(True)
            name=Names[first_match_index]
            userName=name
        top=top*4
        right=right*4
        bottom=bottom*4
        left=left*4
        cv2.rectangle(frame,(left,top),(right, bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-6),font,.75,(0,0,255),2)
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
 
cam.release()
cv2.destroyAllWindows()

#%%
#-------object detection using live camera ------

import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera()
display = jetson.utils.glDisplay()

while display.IsOpen():
	img, width, height = camera.CaptureRGBA()
	detections = net.Detect(img, width, height)
	display.RenderOnce(img, width, height)
	display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
#%%    
#------object detection using images--------------

import jetson.inference
import jetson.utils

import argparse
import sys
import pandas as pd 
import csv

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-inception-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
detectionIDs = list()
# process frames until the user exitsdetectionArray =  list()
while True:
	# capture the next image
	img = input.Capture()

	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=opt.overlay)
	

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))
    
	for detection in detections:
		print(detection)

	# render the image
	output.Render(img)


	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	net.PrintProfilerTimes()

    for i in range(len(detections)):
        detectionIDs.append(detections[i].ClassID)
	

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break
		

#__________________________________________________

#-- write Interest in user file based on object detection 

#get object's index from it's ID
for i in range(len(detectionIDs)):
    index1=0
    index2=1
    for col in data6.index:
        if col == detectionIDs[i]:
            break
        if index1 <= len(data6):
            index1+=1 

#get object's name from index 
    #label=object.iloc[index1, 1]
    category=object.iloc[index1,2]
    if category!="person":
        categories.append(category)

    for col in data5.index:
        if col == 'Ahmed':
            break
        if index2 <= len(data5):
            index2+=1
#write category in user's intrest column 
f = open('Users.csv', 'r')
reader = csv.reader(f)
mylist = list(reader)
f.close()
print(categories)
mylist[index2][6] = categories
my_new_list = open('Users.csv', 'w', newline = '')
csv_writer = csv.writer(my_new_list)
csv_writer.writerows(mylist)
my_new_list.close()

#%% 
#Recommendation based on distance 
distance = pd.DataFrame(data, columns=['ID','Distance','Name'])
# Sorting and dropping the duplicates
sort=distance.sort_values('Distance', ascending=True).drop_duplicates().head(10)
print(sort)



#%%
#Recommendation based on history 
desc_tfidf = TfidfVectorizer(stop_words='english')
# filling the missing values with empty string
desc['Description'] = desc['Description'].fillna('')
# computing TF-IDF matrix required for calculating cosine similarity
description_matrix = desc_tfidf.fit_transform(desc['Description'])
# Let's check the shape of computed matrix
description_matrix.shape
# computing cosine similarity matrix using linear_kernal of sklearn
cosine_similarity = linear_kernel(description_matrix, description_matrix)
indices = pd.Series(desc['Name'].index)

def recommend(index, cosine_sim=cosine_similarity):
    id = indices[index]
    # Get the pairwsie similarity scores of all books compared to that book, 
    # sorting them and getting top 5
    similarity_scores = list(enumerate(cosine_sim[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:2]
    # Get the books index
    index = [i[0] for i in similarity_scores]
    # Return the top 5 most similar books using integer-location based indexing (iloc)
    return desc['Name'].iloc[index]

history = data2.loc[userName]["History"]
place = data3.loc[history]["Name"]
print(place)

index = 0 # history place's index
# iterating over indices
for col in data4.index:
    if col == place:
        break
    if index < len(data4):
        index+=1
#index of history place
#print("place index",index)
#print(len(data4))

if index == len(data4):
    print("not found")
else:
   recoomenderPlaces =recommend(index)
   #print(recoomenderPlaces)
   #-------------end of recommendation based on history----------

   # index of recommender place 
   index3=0
   for col in data4.index:
    if col == recoomenderPlaces:
        break
    if index3 <= len(data4):
        index3+=1
        print (index3)
        placeID=desc.iloc[index3, 0]
        print("place id ",placeID)

        index4=1 # user index

    for col in data2.index:
        if col == userName:
            break
        if index4 <= len(data2):
          index4+=1

    f = open('Users.csv', 'r')
    reader = csv.reader(f)
    mylist = list(reader)
    f.close()
    mylist[index4][5] = placeID # write place ID of what we recommend in user's history 
    my_new_list = open('Users.csv', 'w', newline = '')
    csv_writer = csv.writer(my_new_list)
    csv_writer.writerows(mylist)
    my_new_list.close()

#%%
#Recommendation based on similar keyword between user's intrest and Neom Places 

interest = users.iloc[index4, 6] #[user's index , intrest column No. ] 6= intrest in users tables 
interestxArray = interest.split(",")
place = pd.read_csv("descrption.csv", index_col="Description")# what column we want similraty based on 
placesArray=[]
index5=0
for col in place.index5:
    if index5 < len(place):
        placesArray.append(col)    

#--------------------------------------------------------

interestList = interestxArray
PlacesList = placesArray

# Create an instance of the class. This is otherwise called as an object 
csObj = cs(interestList,PlacesList)   
#csObj.dissimilar_input_items(similarity_threshold = 0.65)

# csObj is now the object of Calculate Similarity class. 
csObj.fuzzy_match_output(output_csv_name = 'Recommendation Places Output.csv', output_csv_path = r'C:\Users\RanaA\PycharmProjects\pythonProject')
