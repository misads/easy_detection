# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import os,sys
#import cPickle
import _pickle as cPickle
import numpy as np
from scripts.eval_ap import parse_rec
from scripts.eval_all import get_image_xml_name
from utils import load_class_names
from scripts.my_eval import compute_ap
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

_classes = None
def convert_bb2lab(classname, imagepath):
    info_path = imagepath.replace('images', 'labels'). \
            replace('JPEGImages', 'labels'). \
            replace('.jpg', '.txt').replace('.png','.txt')
    img = Image.open(imagepath)
    w, h = img.size
    objs = []
    try:
        gt_bbs = np.loadtxt(info_path)
    except:
        return objs

    gt_bbs = gt_bbs.reshape(gt_bbs.size//5, 5)
    for i in range(len(gt_bbs)):
        obj = {}
        gt_bb = gt_bbs[i]

        obj['name'] = classname[(int)(gt_bb[0])]
        obj['pose'] = 'Unspecified'
        obj['truncated'] = 0
        obj['difficult'] = 0

        bb = np.zeros(4);
        hbw = gt_bb[3]/2.0 # half bounding box width
        hbh = gt_bb[4]/2.0 # half bounding box height
        bb[0] = (int)((gt_bb[1] - hbw) * w)    # xmin
        bb[1] = (int)((gt_bb[2] - hbh) * h)    # ymin
        bb[2] = (int)((gt_bb[1] + hbw) * w)    # xmax
        bb[3] = (int)((gt_bb[2] + hbh) * h)    # ymax
        obj['bbox'] = bb
        objs.append(obj)
    return objs

def coco_eval(detpath, imagesetfile, classname, cachedir,
             ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = coco_eval(detpath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            imagekey = os.path.basename(imagename).split('.')[0]    
            lab = convert_bb2lab(_classes, imagename)
            if len(lab) > 0:
                recs[imagekey] = lab
            else:
                print("skipped key: {}, path: {}".format(imagekey, imagename))

            if i % 100 == 0:
                print ('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print ('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        imagekey = os.path.basename(imagename).split('.')[0]
        try:
            R = [obj for obj in recs[imagekey] if obj['name'] == classname]
        except KeyError:
            #print("skipped: %s %s" % (imagename, imagekey))
            continue;
        except Exception as e:
            print(type(e))
            print(e.args)
            print(e)
            print("%s %s" % (imagename, imagekey))
            exit(0)

        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagekey] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        try:
            R = class_recs[image_ids[d]]
        except KeyError:
            #print("skipeed: {}".format(image_ids[d]))
            continue;

        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = compute_ap(rec, prec, use_07_metric)

    #print('class: {:<10s} \t num occurrence: {:4d}'.format(classname, npos))

    return rec, prec, ap, npos
    
def _do_python_eval(res_prefix, imagesetfile, classesfile, output_dir = 'output'):
    
    filename = res_prefix + '{:s}.txt'

    cachedir = os.path.join(output_dir, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = False
    #print ('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    global _classes
    _classes = load_class_names(classesfile)

    total = 0
    for i, cls in enumerate(_classes):
        if cls == '__background__':
            continue
      
        rec, prec, ap, noccur = coco_eval(
            filename, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        total += noccur
        print('AP for {:<10s} = {:.4f} with {:4d} views'.format(cls, ap, noccur))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    print('Mean AP = {:.4f} with total {:4d} views'.format(np.mean(aps), total))
    
    print('~'*30)
    print(' '*10, 'Results:')
    print('-'*30)
    for i, ap in enumerate(aps):
        print('{:<10s}\t{:.3f}'.format(_classes[i], ap))
    print('='*30)
    print('{:^10s}\t{:.3f}'.format('Average', np.mean(aps)))
    print('~'*30)
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')


if __name__ == '__main__':
    #res_prefixc = '/data/hongji/darknet/results/comp4_det_test_' 
    #res_prefix = 'results/comp4_det_test_'    
    #test_file = 'data/sketch_test.txt'
    #class_names = 'data/sketch.names'
    res_prefix = sys.argv[1]
    test_file = sys.argv[2]
    class_names = sys.argv[3]
    _do_python_eval(res_prefix, test_file, class_names, output_dir = 'output')


