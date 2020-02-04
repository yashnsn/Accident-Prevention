# Import socket module
import socket, pickle, cv2, numpy, time           
 
# Create a socket object
j = [] 
# Define the port on which you want to connect
port = 12345               
 
# connect to the server on local computer

while True:
	#for j in range(1,31):
	str2 = str(7) + ".avi"

	
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(str2, fourcc, 30, (512, 512))

	for i in range(1,30):
		s = socket.socket()    
		s.connect(('192.168.100',port))
		img = numpy.zeros((512,512,3), numpy.uint8)
		data = s.recv(1024)
		j = pickle.loads(data)
		#print(j)
		#str1 = str(i) + ".jpeg"
		for d in j:
			cv2.circle(img,d,1,(0,0,255),1)
			#cv2.namedWindow('img',cv2.WINDOW_NORMAL)
		#cv2.imshow('video',img)
		# close the connection 
		out.write(img)
		s.close() 
		#time.sleep(10)
	video_src = '7.avi'
	cap = cv2.VideoCapture(video_src)
	while True:
		ret, img = cap.read()
		if (type(img) == type(None)):
			break
		cv2.imshow('video', img)
		if cv2.waitKey(33) == 27:
			break

	cv2.destroyAllWindows()
		
