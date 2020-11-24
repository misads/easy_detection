import torch
from darknet import Darknet
import dataset
from torchvision import datasets, transforms
from utils import get_all_boxes, bbox_iou, nms, read_data_cfg, load_class_names
from image import correct_yolo_boxes
import os
import pdb

def valid(datacfg, cfgfile, weightfile, outfile):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    name_list = options['names']
    prefix = 'results'
    names = load_class_names(name_list)

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    m.cuda()
    m.eval()

    valid_dataset = dataset.listDataset(valid_images, shape=(m.width, m.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 2
    assert(valid_batchsize > 1)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    fps = [0]*m.num_classes
    if not os.path.exists('results'):
        os.mkdir('results')
    for i in range(m.num_classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
        fps[i] = open(buf, 'w')
   
    lineId = -1
    
    conf_thresh = 0.005
    nms_thresh = 0.45
    if m.net_name() == 'region': # region_layer
        shape=(0,0)
    else:
        shape=(m.width, m.height)
    for _, (data, target, org_w, org_h) in enumerate(valid_loader):
        data = data.cuda()
        output = m(data)
        batch_boxes = get_all_boxes(output, shape, conf_thresh, m.num_classes, only_objectness=0, validation=True)
        
        for i in range(len(batch_boxes)):
            lineId += 1
            fileId = os.path.basename(valid_files[lineId]).split('.')[0]
            #width, height = get_image_size(valid_files[lineId])
            width, height = float(org_w[i]), float(org_h[i])
            print(valid_files[lineId])
            boxes = batch_boxes[i]
            correct_yolo_boxes(boxes, width, height, m.width, m.height)
            boxes = nms(boxes, nms_thresh)
            for box in boxes:
                x1 = (box[0] - box[2]/2.0) * width
                y1 = (box[1] - box[3]/2.0) * height
                x2 = (box[0] + box[2]/2.0) * width
                y2 = (box[1] + box[3]/2.0) * height

                det_conf = box[4]
                for j in range((len(box)-5)//2):
                    cls_conf = box[5+2*j]
                    cls_id = int(box[6+2*j])
                    prob = det_conf * cls_conf
                    fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))

    for i in range(m.num_classes):
        fps[i].close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]
        outfile = 'comp4_det_test_'
        valid(datacfg, cfgfile, weightfile, outfile)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
