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

def pavel_fce():
  return
