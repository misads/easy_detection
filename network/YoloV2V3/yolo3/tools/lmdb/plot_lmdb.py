import lmdb
import cv2
import numpy as np

env = lmdb.open('data/face_test_lmdb',
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)

with env.begin(write=False) as txn:
    nSamples = int(txn.get('num-samples'))
    #print nSamples
    for index in range(nSamples):
        image_key = 'image-%09d' % (index+1)
        label_key = 'label-%09d' % (index+1)
        imageBin = txn.get(image_key)
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
        imgH, imgW = img.shape[0], img.shape[1]
        labels = txn.get(label_key).rstrip().split('\n')
        for label in labels:
            label = label.split()
            box = [float(i) for i in label]
            x = box[1]*imgW
            y = box[2]*imgH
            w = box[3]*imgW
            h = box[4]*imgH
            x1 = int(x - w/2.0)
            x2 = int(x + w/2.0)
            y1 = int(y - h/2.0)
            y2 = int(y + h/2.0)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)
        savename = 'tmp/%s.png'%(image_key)
        print('save %s' % (savename))
        cv2.imwrite(savename, img)

