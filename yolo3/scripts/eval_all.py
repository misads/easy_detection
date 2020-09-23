import os
import os.path
from PIL import Image
import sys
from torch.autograd import Variable
from darknet import Darknet
from utils import get_all_boxes, do_detect, plot_boxes, load_class_names, image2torch, get_region_boxes, nms
import numpy as np

conf_thresh = 0.005
#conf_thresh = 0.5
nms_thresh = 0.45
def save_boxes(imgfile, img, boxes, savename):
    fp = open(savename, 'w')
    filename = os.path.basename(savename)
    filename = os.path.splitext(filename)[0]
    fp.write('# imagepath = %s\n' % imgfile)
    fp.write('# basename = %s\n' % filename)
    fp.write('# nbbs = %d\n' % len(boxes))
    width = img.width
    height = img.height
    # box[0], box[1] : center x, center y
    # box[2], box[3] : width, height
    # box[4] : confidence
    # box[5] : max confidence of the class
    # box[6] : max class id
    for box in boxes:
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        det_conf = box[4]
        for j in range((len(box)-5)//2):
            cls_conf = box[5+2*j]
            cls_id = box[6+2*j]
            prob = det_conf * cls_conf
            fp.write('%d %f %f %f %f %f\n' % (cls_id, prob, x1, y1, x2, y2 ))
    fp.close()

def get_det_image_name(imagefile):
    file, ext = os.path.splitext(imagefile)
    imgname = file + "_det" + ext
    return imgname

def get_det_result_name(imagefile):
    return imagefile.replace('images', 'results').replace('JPEGImages', 'results').replace('.jpg', '.det').replace('.png','.det')

def get_image_xml_name(imagefile):
    return imagefile.replace('images', 'Annotations').replace('JPEGImages', 'Annotations').replace('.jpg', '.xml').replace('.png','.xml')

def eval_list(cfgfile, namefile, weightfile, testfile):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    use_cuda = 1
    if use_cuda:
        m.cuda()

    class_names = load_class_names(namefile)

    file_list = []
    with open(testfile, "r") as fin:
        for f in fin:
            file_list.append(f.strip())

    for imgfile in file_list:
        img = Image.open(imgfile).convert('RGB')
        sized = img.resize((m.width, m.height))
        filename = os.path.basename(imgfile)
        filename = os.path.splitext(filename)[0]
        #print(filename, img.width, img.height, sized_width, sized_height)

        if m.width * m.height > 1024 * 2560:
            print('omit %s' % filename)
            continue

        if False:
            boxes = do_detect(m, sized, conf_thresh, nms_thresh, use_cuda)
        else:
            m.eval()
            sized = image2torch(sized).cuda();
            #output = m(Variable(sized, volatile=True)).data
            output = m(sized)
            #boxes = get_region_boxes(output, conf_thresh, m.num_classes, m.anchors, m.num_anchors, 0, 1)[0]
            boxes = get_all_boxes(output, conf_thresh, m.num_classes)[0]
            boxes = np.array(nms(boxes, nms_thresh))

        if False:
            savename = get_det_image_name(imgfile)
            print('img: save to %s' % savename)
            plot_boxes(img, boxes, savename, class_names)

        if False:
            savename = get_det_result_name(imgfile)
            print('det: save to %s' % savename)
            save_boxes(imgfile, img, boxes, savename)

if __name__ == '__main__':
    savedir = None
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1]
        namefile = sys.argv[2]
        wgtfile = sys.argv[3]
        testlist = sys.argv[4]

        eval_list (cfgfile, namefile, wgtfile, testlist)
    else:
        print("Usage: %s cfgfile classname weight testlist" % sys.argv[0] )

