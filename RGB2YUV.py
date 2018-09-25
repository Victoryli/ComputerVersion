import time
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

#imgname1 = './hand_x.jpg'
#imgname2 = './hand_xz.jpg'

class HandDetect:
    def __init__ (self):
        self.crl = 133
        self.crh = 173
        self.cbl = 77
        self.cbh = 127
    def ResizeIMAGE(self,IMAGE,SIZE):
        return cv2.resize(IMAGE,SIZE,interpolation=cv2.INTER_CUBIC)

    def image_YUV_progress (self,IMAGE):

       # img = cv2.imread(self.impath1)
       # res = cv2.resize(img,(240,200),interpolation=cv2.INTER_CUBIC)
        yuv = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2YCR_CB) #BGR 2 YCrCb
        res = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2RGB)
        skin = np.zeros(IMAGE.shape, np.uint8)
        skin = res.copy()

        #img2 = cv2.imread(imgname2)
        #res2 = cv2.resize(img2,(240,200),interpolation=cv2.INTER_CUBIC)
        #yuv2 = cv2.cvtColor(res2, cv2.COLOR_BGR2YCR_CB) #BGR 2 YCrCb
        #res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)
        #skin2 = np.zeros(res2.shape, np.uint8)
        #skin2 = res2.copy()


#def Thresh_YUV_Image(YUVimage,RGBimage,crL,crH,cbL,cbH):
        imageShape = yuv.shape
        print (imageShape)
       # threshImage = np.zeros(imageShape)
        print (imageShape[0])
        for i in range(imageShape[0]):
            for j in range(imageShape[1]):
                if ((yuv[i][j][1] >= self.crl) and (yuv[i][j][1] <= self.crh) and (yuv[i][j][2] >= self.cbl) and (yuv[i][j][2] <= self.cbh)):
                    hand = 1
                else :
               # hand = 0
                    skin[i][j][0] = 0
                    skin[i][j][1] = 0
                    skin[i][j][2] = 0
                

    #return threshImage.astype(int)
        return cv2.cvtColor(skin, cv2.COLOR_RGB2BGR)

   # def Image_RGB2BGR(IMAGE_RGB2BGR):
   #         return cv2.cvtColor(IMAGE_RGB2BGR, cv2.COLOR_RGB2BGR)

#Thresh1 = Thresh_YUV_Image(yuv1,skin1,133,173,77,127)
#Thresh2 = Thresh_YUV_Image(yuv2,skin2,133,173,77,127)

#hand1 = Image_RGB2BGR(Thresh1)
#hand2 = Image_RGB2BGR(Thresh2)

#cv2.imshow("Thresh1",hand1)
#cv2.imshow("Thresh2",hand2)
#cv2.waitKey(0)
#time.sleep(3)

#####################################################
#Find rectangles

#####################################################
    def FindBox(self,IMAGE,IMAGE_YUV):
        imgray = cv2.cvtColor(IMAGE_YUV,cv2.COLOR_BGR2GRAY)
        print (imgray.shape)
        ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        #cv2.drawContours(hand1,contours,-1,(0,0,255),3)
        for m in range(0,len(contours)): 
            x, y, w, h = cv2.boundingRect(contours[m])
            if (w>20 and h>20):  
                cv2.rectangle(IMAGE, (x,y), (x+w,y+h), (153,153,0), 2)
        return IMAGE
######################################################

#handBox1 = findBox(hand1)
#handBox2 = findBox(hand2)
#cv2.imshow("handBox1",handBox1)
#cv2.imshow("handBox2",handBox2)
#cv2.waitKey(0)

####################################################3
#############################################################################

#SIFT

#############################################################################
    def SIFTprogress (self,IMAGE_YUV1,IMAGE_YUV2,TIME):
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(IMAGE_YUV1,None)   #des是描述子
        kp2, des2 = sift.detectAndCompute(IMAGE_YUV2,None)  #des是描述子

        hmerge = np.hstack((IMAGE_YUV1, IMAGE_YUV2)) #水平拼接
        cv2.imshow("Thresh", hmerge) #拼接显示
       # cv2.waitKey(0)
        time.sleep(TIME)

        img3 = cv2.drawKeypoints(IMAGE_YUV1,kp1,res1,color=(255,0,255)) #画出特征点，并显示为红色圆圈
        img4 = cv2.drawKeypoints(IMAGE_YUV2,kp2,res2,color=(255,0,255)) #画出特征点，并显示为红色圆圈
        hmerge = np.hstack((img3, img4)) #水平拼接
        cv2.imshow("point",  cv2.cvtColor(hmerge, cv2.COLOR_RGB2BGR)) #拼接显示为gray
       # cv2.waitKey(0)
        time.sleep(TIME)

        # BFMatcher解决匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        # 调整ratio
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        img5 = cv2.drawMatchesKnn(res1,kp1,res2,kp2,good,None,flags=2)
        cv2.imshow("BFmatch", cv2.cvtColor(img5, cv2.COLOR_RGB2BGR))
       # cv2.waitKey(0)
        time.sleep(TIME)
#cv2.destroyAllWindows()
if __name__ =='__main__':

    imgpath1 = './hand_x.jpg'
    imgpath2 = './hand_xz.jpg'
    imageONE = cv2.imread(imgpath1)
    imageTWO = cv2.imread(imgpath2)
    HD = HandDetect()
    res1 = HD.ResizeIMAGE(imageONE,(240,200))
    res2 = HD.ResizeIMAGE(imageTWO,(240,200))
    yuv1 = HD.image_YUV_progress(res1)
    yuv2 = HD.image_YUV_progress(res2)
    Box1 = HD.FindBox(res1,yuv1)
    Box2 = HD.FindBox(res2,yuv2)
    cv2.imshow("box",Box1)
   # cv2.imwrite("1.jpg",Box1, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.waitKey(0)
    sift1 = HD.SIFTprogress(yuv1,yuv2,2)   
