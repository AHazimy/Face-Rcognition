# Importing all necessary libraries
import cv2
import os

from api import face_extractor

## for training data
# try:
# 	if not os.path.exists('data/train'):
# 		os.makedirs('data/train')
# except OSError:
# 	print ('Error: Creating directory of data')
 
 
## for testing data
try:
	if not os.path.exists('data/test_model'):
		os.makedirs('data/test_model')
except OSError:
	print ('Error: Creating directory of data')


data_path = './db_test_real/'
videos = os.listdir(data_path)

client_label = ""
count = 0

for video in videos:
    
	client_label_temp = f'data/test_model/{video.split("_")[0]}'
	
	if client_label_temp != client_label:
		count = 0
    
	client_label = client_label_temp
    
	if not os.path.exists(client_label):
		os.makedirs(client_label)

	cam = cv2.VideoCapture(os.path.join(data_path,video))

	

	while(True):
	
		ret,frame = cam.read()

		if ret:
			if face_extractor(frame) is not None:
				count+=1
				face = cv2.resize(face_extractor(frame), (200,200))
				face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)    
				file_name_path = f'{os.path.join(client_label, str(count))}.jpg'
				cv2.imwrite(file_name_path, face)
				cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
				# cv2.imshow('Face Cropper', face)
				print(f"Image {count} for the client {client_label}")
    
			else:
				print("Face not found")
				pass
		else:
			cam.release()
			cv2.destroyAllWindows()
			break






