import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

mouth_cascade=cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")


org=(30,30)
fontFace=cv2.FONT_HERSHEY_SIMPLEX
fontScale=1
cap=cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        print("Maske")
        
        
    gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    
    faces=face_cascade.detectMultiScale(gray_image,1.1,15)
    
    i=0#Tek ağız karesi
    if(len(faces)==0):
        cv2.putText(frame,"Yuz Tespit Edilemedi",org,fontFace,fontScale,(255,255,255),2)
    
    else:
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#Yüzü kare içerisine al
            roi_gray=gray_image[y:y+h,x:x+w]
            
            mouth=mouth_cascade.detectMultiScale(roi_gray,1.4,15)
            
            if(len(mouth)==0):
               cv2.putText(frame,"Maske Tespit Edildi",org,fontFace,fontScale,(0,255,0),2,cv2.LINE_AA)
            
            else:
               cv2.putText(frame,"Maske Tespit Edilemedi",org,fontFace,fontScale,(0,0,255),2,cv2.LINE_AA)
               #Ağız koordinat
               for mx,my,mw,mh in mouth:
                   if i==0:
                       i+=1
                       cv2.rectangle(frame,(mx+x,my+y),(mx+mw+x,my+mh+y),(255,255,0),2)#Yüzü kare içerisine al
                   else:
                       pass
                    
    
    
    cv2.imshow("Maske Tespiti",frame)
    
    if cv2.waitKey(1)&0xFF==ord("q"):
        print("Iyi Gunler..")
        break

cap.release()
cv2.destroyAllWindows()






