import cv2

def From_Image():

   image = cv2.imread("C:\\Users\\HP\\Pictures\\Saved Pictures\\bmw.jfif")
   image = cv2.resize(image , (800 , 600))
   file_Objects_path = "coco.names"
   Objects = []

   with open(file_Objects_path , 'r') as file :
      Objects = file.read().rstrip('\n').split('\n')
               
   Algorithm_1 = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
   Algorithm_2 = "frozen_inference_graph.pb"

   net = cv2.dnn_DetectionModel(Algorithm_1 , Algorithm_2)
   net.setInputSize(320 , 230)
   net.setInputScale(1.0 / 127.5)
   net.setInputMean((127.5, 127.5, 127.5))
   net.setInputSwapRB(True)


   Objects_IDs , confidences , b_box = net.detect(image , confThreshold = 0.5) 

   for Objects_ID , confidence , box in zip(Objects_IDs , confidences , b_box) :
      cv2.rectangle(image , box , color=(0,255,0) , thickness = 2)
      cv2.putText(image , Objects[Objects_ID-1] ,(box[0] + 10, box[1] + 20) ,
                  cv2.FONT_HERSHEY_COMPLEX , 1 , (255,0,0) , thickness = 2)
     
   cv2.imshow("Photo" , image)
   if cv2.waitKey(0) == ord('q') :
      cv2.destroyAllWindows()



def From_camera():
   camera = cv2.VideoCapture(0)

   file_Objects_path = "coco.names"
   Objects_names = []
   with open(file_Objects_path , 'r') as file:
       Objects_names = file.read().split('\n')

   Algorithm_1 = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtx"
   Algorithm_2 = "frozen_inference_graph.pb"

   net = cv2.dnn_DetectionModel(Algorithm_1 , Algorithm_2)
   net.setInputSize(320 , 230)
   net.setInputScale(1.0 / 127.5)
   net.setInputMean((127.5, 127.5, 127.5))
   net.setInputSwapRB(True)


   while True :
      sucess , cam = camera.read()

      Object_IDs , confidences , b_box = net.detect(cam , confThreshold = 0.5)
   
      if len(Object_IDs) != 0 :
         for Object_ID , confidence , box in  zip(Object_IDs , confidences , b_box):
            cv2.rectangle(cam , box , (0,255,0) , thickness = 1)
            cv2.putText(cam , Objects_names[Object_ID-1]  , (box[0] + 10 , box[1] + 20) , 
                        cv2.FONT_HERSHEY_COMPLEX , 1 , (255,0,0) , thickness = 2)  
   
   
      cv2.imshow("photo" , cam)
      if cv2.waitKey(1) == ord('q'):
         break


# just call the functions
From_Image()
# From_camera()

