def najdi_brouka(image):
    import scipy
    import skimage
    import skimage.io
    import skimage.morphology
    import matplotlib.pyplot as plt
    import cv2
    from skimage.filters import threshold_otsu, threshold_local, threshold_multiotsu
    import numpy as np
    from skimage import morphology
    import skimage.measure
    import copy
    
#     URL = 'D:\\ZČU\\ZDO\\Varroaza\\images\\Original_616_image.jpg'
#     imrgb = skimage.io.imread(URL)

    # konverze obrazku do cernobile pro dalsi zpracovani
    imrgb = image
    img = cv2.cvtColor(imrgb, cv2.COLOR_BGR2GRAY)
    
    # urceni spodni a horni meze pro kazdou slozku z RGB, filtrace ostatnich barev
    lower_color = np.array([0,0,0])
    upper_color = np.array([80,80,60])
    mask = cv2.inRange(imrgb, lower_color, upper_color)
    
    # morfologicke operace
    mask = scipy.ndimage.morphology.binary_opening(mask,iterations=2)
    mask = scipy.ndimage.morphology.binary_closing(mask,iterations=3)
    mask = scipy.ndimage.morphology.binary_dilation(mask,iterations=3)
    mask = scipy.ndimage.morphology.binary_erosion(mask,iterations=2)
    mask = mask.clip(max=1)


#     plt.imshow(mask)
#     skimage.io.imsave("mask.jpg", mask)

    # Otsuovo prahovani na cernobily obrazek, 4 skupiny, pouziti te nejtmavsi
    thresh = threshold_multiotsu(img, 4)
    otsu = img < thresh[0]
    
    # morfologicke operace se zbytkem po Otsuovi
    morf_otsu = scipy.ndimage.morphology.binary_closing(otsu,iterations=3) # mozna
    morf_otsu = scipy.ndimage.morphology.binary_erosion(morf_otsu,iterations=1)
    morf_otsu = morf_otsu.astype(int)

    # spojeni cernobile a barevne masky
    kombi = np.multiply(mask, morf_otsu)
    
    # labeling spojene masky
    all_labels = morphology.label(kombi).astype(int)
    
    # ziskani informaci o jednotlivych regionech
    props = skimage.measure.regionprops(all_labels)
    
    # priprava poli pro ulozeni zajimavych regionu, nejdrive filtrace pomoci obsahu, pote podle ostatnich vlastnosti
    Kandidati = []
    Klestici = []
    
    # filtrace podle oblasti
    for prop in props:
        if prop.area > 120 and prop.area < 350:
            Kandidati.append(prop)
            
    # filtrace podle vlastnosti, ktere jsou charakteristicke pro klestiky
    for k in Kandidati:
        if k.eccentricity > 0.4 and k.eccentricity < 0.85: #< ------- kulaty ma 0.4, klidne az k 0.8
            if k.major_axis_length > 15 and k.major_axis_length < 30: #<------------ zatim 19 az 23
                if k.minor_axis_length > 11 and k.minor_axis_length < 18: #<------------ zatim 14 az 18
                    if k.perimeter > 35 and k.perimeter < 110: #<------------ mozna mensi spodni mez, mozna vetsi horni mez
                        Klestici.append(k)
                        
         
        
        
    vysledek = np.ascontiguousarray(imrgb)
    maska = np.zeros_like(img)
    for klestik in Klestici:
        y = (int(klestik.centroid[0])-10, int(klestik.centroid[0])+10)
        x = (int(klestik.centroid[1])-10, int(klestik.centroid[1])+10)

        for j in range(len(klestik.coords)):
            bod_y = klestik.coords[j][0]
            bod_x = klestik.coords[j][1]
            maska[bod_y][bod_x] = 255
#         vysledek = cv2.rectangle(vysledek, (x[0],y[0]), (x[1],y[1]), (255,0,0), 2)
        
    
    return maska

def pavel_detector(img_rgb):
    import os
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    import skimage.io
    
    counter = 0
    #img_rgb = cv2.imread('ZDO_data\\images\\Original_608_image.jpg', cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h_img, w_img = img_gray.shape[::]
    mask = np.zeros((h_img,w_img))
    for name in os.listdir('ZDO_data\\templates'):
        #if counter%10==0:
        #    print(counter)

        template = cv2.imread('ZDO_data\\templates\\'+name,0)
        h, w = template.shape[::]

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        #plt.imshow(res, cmap='gray')

        threshold = 0.8 #Pick only values above 0.8. For TM_CCOEFF_NORMED, larger values = good fit.

        loc = np.where( res >= threshold)  
        #Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.

        #Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
        #then the second item and then third, etc. 

        for pt in zip(*loc[::-1]):   #-1 to swap the values as we assign x and y coordinate to draw the rectangle. 
            #Draw rectangle around each object. We know the top left (pt), draw rectangle to match the size of the template image.
            #cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)  #Red rectangles with thickness 2. 
            mask[pt[1]:pt[1]+w, pt[0]:pt[0]+h]=1
            
        counter +=1
    #cv2.imwrite('ZDO_data\\result.jpg', img)
    return mask
