def najdi_brouka(imrgb)
#     URL = 'D:\\ZČU\\ZDO\\Varroaza\\images\\Original_608_image.jpg'
    imrgb = skimage.io.imread(URL)
    img = cv2.cvtColor(imrgb, cv2.COLOR_BGR2GRAY)
    plt.imshow(imrgb, cmap="gray")
    
    lower_color = np.array([0,0,0])
    upper_color = np.array([70,80,60])
    #upper_color = np.array([255,255,255])
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.inRange(imrgb, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # block_size = 105
    # binary_adaptive = img < threshold_local(img, block_size, offset=0)
    thresh = threshold_multiotsu(img, 4)
    # otsu1 = img > thresh[0]
    otsu = img < thresh[0]
    # otsu = np.multiply(otsu1, otsu2) 
    # plt.imshow(image, cmap = "gray")
    # skimage.io.imsave("otsu1.jpg", otsu1)
    # skimage.io.imsave("otsu2.jpg", otsu2)
    # skimage.io.imsave("otsu.jpg", otsu)
    otevreni = scipy.ndimage.morphology.binary_closing(otsu,iterations=2)
    otevreni = scipy.ndimage.morphology.binary_erosion(otevreni,iterations=2)
    otevreni = otevreni.astype(int)
    mask = mask.clip(max=1)
    kombi = np.multiply(mask, otevreni)
    plt.imshow(kombi)
    all_labels = morphology.label(kombi).astype(int)
    # all_labels = morphology.label(mask).astype(int)
    props = skimage.measure.regionprops(all_labels)
    plt.imshow(all_labels)
    
    Kandidati = []
    Klestici = []
    for prop in props:
        if prop.area > 120 and prop.area < 350:
            Kandidati.append(prop)

    for k in Kandidati:
        if k.eccentricity > 0.4 and k.eccentricity < 0.85: #< ------- kulaty ma 0.4, klidne az k 0.8
            if k.major_axis_length > 15 and k.major_axis_length < 30: #<------------ zatim 19 az 23
                if k.minor_axis_length > 11 and k.minor_axis_length < 18: #<------------ zatim 14 az 18
                    if k.perimeter > 35 and k.perimeter < 110: #<------------ mozna mensi spodni mez, mozna vetsi horni mez
                        Klestici.append(k)
                        
    lul = np.ascontiguousarray(imrgb)
    for klestik in Klestici:
        y = (int(klestik.centroid[0])-10, int(klestik.centroid[0])+10)
        x = (int(klestik.centroid[1])-10, int(klestik.centroid[1])+10)
        print((x[0],y[0]), (x[1],y[1]))
        print(klestik.perimeter, klestik.eccentricity, klestik.major_axis_length, klestik.minor_axis_length, klestik.area)
        lul = cv2.rectangle(lul, (x[0],y[0]), (x[1],y[1]), (255,0,0), 2)
    skimage.io.imsave("vysledek.jpg", lul)
    
    
    # return maska 
    return

def pavel_detector(img_rgb):

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
