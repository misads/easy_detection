#!/usr/bin/python
# encoding: utf-8
import os
from PIL import Image, ImageFile
import numpy as np

# to avoid image file truncation error
ImageFile.LOAD_TRUNCATED_IMAGES = True

def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def image_scale_and_shift(img, new_w, new_h, net_w, net_h, dx, dy):
    scaled = img.resize((new_w, new_h))
    # find to be cropped area
    sx, sy = -dx if dx < 0 else 0, -dy if dy < 0 else 0
    ex, ey = new_w if sx+new_w<=net_w else net_w-sx, new_h if sy+new_h<=net_h else net_h-sy
    scaled = scaled.crop((sx, sy, ex, ey))

    # find the paste position
    sx, sy = dx if dx > 0 else 0, dy if dy > 0 else 0
    assert sx+scaled.width<=net_w and sy+scaled.height<=net_h
    new_img = Image.new("RGB", (net_w, net_h), (127, 127, 127))
    new_img.paste(scaled, (sx, sy))
    del scaled
    return new_img

def image_scale_and_shift_nosafe(img, new_w, new_h, net_w, net_h, dx, dy):
    scaled = img.resize((new_w, new_h))
    new_img = Image.new("RGB", (net_w, net_h), (127, 127, 127))
    new_img.paste(scaled, (dx, dy))
    del scaled
    return new_img

def image_scale_and_shift_slow(img, new_w, new_h, net_w, net_h, dx, dy):
    scaled = np.array(img.resize((new_w, new_h)))
    # scaled.size : [height, width, channel]
    
    if dx > 0: 
        shifted = np.pad(scaled, ((0,0), (dx,0), (0,0)), mode='constant', constant_values=127)
    else:
        shifted = scaled[:,-dx:,:]

    if (new_w + dx) < net_w:
        shifted = np.pad(shifted, ((0,0), (0, net_w - (new_w+dx)), (0,0)), mode='constant', constant_values=127)
               
    if dy > 0: 
        shifted = np.pad(shifted, ((dy,0), (0,0), (0,0)), mode='constant', constant_values=127)
    else:
        shifted = shifted[-dy:,:,:]
        
    if (new_h + dy) < net_h:
        shifted = np.pad(shifted, ((0, net_h - (new_h+dy)), (0,0), (0,0)), mode='constant', constant_values=127)
    #print("scaled: {} ==> dx {} dy {} for shifted: {}".format(scaled.shape, dx, dy, shifted.shape))
    return Image.fromarray(shifted[:net_h, :net_w,:])
  
def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = np.random.uniform(1, s)
    if np.random.randint(2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = np.random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation_crop(img, shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = np.random.randint(-dw, dw)
    pright = np.random.randint(-dw, dw)
    ptop   = np.random.randint(-dh, dh)
    pbot   = np.random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = ow / float(swidth)
    sy = oh / float(sheight)
    
    flip = np.random.randint(2)

    cropbb = np.array([pleft, ptop, pleft + swidth - 1, ptop + sheight - 1])
    # following two lines are old method. out of image boundary is filled with black (0,0,0)
    #cropped = img.crop( cropbb )
    #sized = cropped.resize(shape)

    nw, nh = cropbb[2]-cropbb[0], cropbb[3]-cropbb[1]
    # get the real image part
    cropbb[0] = -min(cropbb[0], 0)
    cropbb[1] = -min(cropbb[1], 0)
    cropbb[2] = min(cropbb[2], ow)
    cropbb[3] = min(cropbb[3], oh)
    cropped = img.crop( cropbb )

    # calculate the position to paste
    bb = (pleft if pleft > 0 else 0, ptop if ptop > 0 else 0)
    new_img = Image.new("RGB", (nw, nh), (127,127,127))
    new_img.paste(cropped, bb)

    sized = new_img.resize(shape)
    del cropped, new_img
    
    dx = (float(pleft)/ow) * sx
    dy = (float(ptop) /oh) * sy

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    # for compatibility to nocrop version (like original version)
    return img, flip, dx, dy, sx, sy 

def data_augmentation_nocrop(img, shape, jitter, hue, sat, exp):
    net_w, net_h = shape
    img_w, img_h = img.width, img.height
        
    # determine the amount of scaling and cropping
    dw = jitter * img_w
    dh = jitter * img_h

    new_ar = (img_w + np.random.uniform(-dw, dw)) / (img_h + np.random.uniform(-dh, dh))
    # scale = np.random.uniform(0.25, 2)
    scale = 1.

    if (new_ar < 1):
        new_h = int(scale * net_h)
        new_w = int(net_h * new_ar)
    else:
        new_w = int(scale * net_w)
        new_h = int(net_w / new_ar)
            
    dx = int(np.random.uniform(0, net_w - new_w))
    dy = int(np.random.uniform(0, net_h - new_h))
    sx, sy = new_w / net_w, new_h / net_h
        
    # apply scaling and shifting
    new_img = image_scale_and_shift(img, new_w, new_h, net_w, net_h, dx, dy)
        
    # randomly distort hsv space
    new_img = random_distort_image(new_img, hue, sat, exp)
        
    # randomly flip
    flip = np.random.randint(2)
    if flip: 
        new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
            
    dx, dy = dx/net_w, dy/net_h
    return new_img, flip, dx, dy, sx, sy 

def fill_truth_detection(labpath, crop, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes,5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2 # center x
            bs[i][2] = (y1 + y2)/2 # center y
            bs[i][3] = (x2 - x1)   # width
            bs[i][4] = (y2 - y1)   # height

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            # when crop is applied, we should check the cropped width/height ratio
            if bs[i][3] < 0.002 or bs[i][4] < 0.002 or \
                (crop and (bs[i][3]/bs[i][4] > 20 or bs[i][4]/bs[i][3] > 20)):
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def letterbox_image(img, net_w, net_h):
    im_w, im_h = img.size
    if float(net_w)/float(im_w) < float(net_h)/float(im_h):
        new_w = net_w
        new_h = (im_h * net_w)//im_w
    else:
        new_w = (im_w * net_h)//im_h
        new_h = net_h
    resized = img.resize((new_w, new_h), Image.ANTIALIAS)
    lbImage = Image.new("RGB", (net_w, net_h), (127,127,127))
    lbImage.paste(resized, \
            ((net_w-new_w)//2, (net_h-new_h)//2, \
             (net_w+new_w)//2, (net_h+new_h)//2))
    return lbImage

def correct_yolo_boxes(boxes, im_w, im_h, net_w, net_h):
    im_w, im_h = float(im_w), float(im_h)
    net_w, net_h = float(net_w), float(net_h)
    if net_w/im_w < net_h/im_h:
        new_w = net_w
        new_h = (im_h * net_w)/im_w
    else:
        new_w = (im_w * net_h)/im_h
        new_h = net_h

    xo, xs = (net_w - new_w)/(2*net_w), net_w/new_w
    yo, ys = (net_h - new_h)/(2*net_h), net_h/new_h
    for i in range(len(boxes)):
        b = boxes[i] 
        b[0] = (b[0] - xo) * xs
        b[1] = (b[1] - yo) * ys
        b[2] *= xs
        b[3] *= ys
    return

def load_data_detection(imgpath, shape, crop, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    if crop:         # marvis version
        img,flip,dx,dy,sx,sy = data_augmentation_crop(img, shape, jitter, hue, saturation, exposure)
    else:            # original version
        img,flip,dx,dy,sx,sy = data_augmentation_nocrop(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath, crop, flip, -dx, -dy, sx, sy)
    return img, label
