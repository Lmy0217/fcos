import torch
import numpy as np 
import matplotlib.pyplot as plt 
import os, math, random
from PIL import Image, ImageDraw
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2



class Dataset_CSV(data.Dataset):
    def __init__(self, root, list_file, name_file, frame_name_file,
                    size=1025, train=True, normalize=True, boxarea_th=35,
                    img_scale_min=0.8, augmentation=None, mosaic=False):
        ''''
        Provide:
        self.fnames:      [fname1, fname2, fname3, ...] # image filename
        self.boxes:       [FloatTensor(N1,4), FloatTensor(N2,4), ...]
        self.labels:      [LongTensor(N1), LongTensor(N2), ...]
        self.LABEL_NAMES: ['background', 'person', 'bicycle', ...] in name_file

        Note:
        - root: folder for jpg images
        - list_file: img_name.jpg ymin1 xmin1 ymax1 xmax1 label1 ... /n
        - name_file: background /n class_name1 /n class_name2 /n ...
        - if not have object -> xxx.jpg 0 0 0 0 0
        - remove box when area < boxarea_th
        - label == 0 indecates background 
        '''
        self.root = root
        self.size = size
        self.train = train
        self.normalize = normalize
        self.boxarea_th = boxarea_th
        self.img_scale_min = img_scale_min
        self.augmentation = augmentation
        self.mosaic = mosaic
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.frame_labels = []
        self.LABEL_NAMES = []
        self.FRAME_NAMES = []
        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)
            for line in lines:
                splited = line.strip().split()
                self.fnames.append(splited[0])
                num_boxes = (len(splited) - 2) // 5
                box = []
                label = []
                for i in range(num_boxes):
                    ymin = splited[1+5*i]
                    xmin = splited[2+5*i]
                    ymax = splited[3+5*i]
                    xmax = splited[4+5*i]
                    c = splited[5+5*i]
                    box.append([float(ymin),float(xmin),float(ymax),float(xmax)])
                    label.append(int(c))
                self.boxes.append(torch.FloatTensor(box))
                self.labels.append(torch.LongTensor(label))
                self.frame_labels.append(torch.tensor([int(splited[-1]) - 1], dtype=torch.long))
        self.weights = 1 / torch.from_numpy(np.bincount(torch.cat(self.frame_labels).numpy()).astype(np.float32))
        self.weights = self.weights / self.weights.sum()
        with open(name_file, encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                self.LABEL_NAMES.append(line.strip())
        with open(frame_name_file, encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                self.FRAME_NAMES.append(line.strip())
        self.normalizer = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    

    def __len__(self):
        return self.num_samples
    

    def __getitem__(self, idx):
        """
        Return:
        img:          FloatTensor(3, size, size)
        boxes:        FloatTensor(box_num, 4)
        labels:       LongTensor(box_num)
        frame_label:  LongTensor(1)
        loc:          FloatTensor(4)
        scale:        float scalar
        """

        if self.mosaic:
            img, boxes, labels, frame_label, loc, scale = self.load_mosaic(idx)
            img = Image.fromarray(img)
            labels = labels.squeeze(-1)
            frame_label = frame_label.squeeze(-1)
            # loc = torch.FloatTensor([min(loc[0, 0], loc[1, 0]), min(loc[0, 1], loc[2, 1]),
            #                  max(loc[2, 2], loc[3, 2]) + self.size // 2, max(loc[1, 3], loc[3, 3]) + self.size // 2])
            loc = torch.FloatTensor([0, 0, self.size, self.size])
            scale = scale.squeeze(-1)
        else:
            # idx = 5
            img, boxes, labels, frame_label, loc, scale = self.load_image(idx, self.size, self.train)
            scale = torch.FloatTensor([scale])

        img = transforms.ToTensor()(img)
        if self.normalize:
            img = self.normalizer(img)

        return img, boxes, labels, frame_label, loc, scale


    def collate_fn(self, data):
        '''
        Return:
        img     FloatTensor(batch_num, 3, size, size)
        boxes   FloatTensor(batch_num, N_max, 4)
        Labels  LongTensor(batch_num, N_max)
        loc     FloatTensor(batch_num, 4)
        scale   FloatTensor(batch_num)
        '''
        img, boxes, labels, frame_label, loc, scale = zip(*data)
        img = torch.stack(img)
        batch_num = len(boxes)
        N_max = 0
        for b in range(batch_num):
            n = boxes[b].shape[0]
            if n > N_max: N_max = n
        boxes_t = torch.zeros(batch_num, N_max, 4)
        labels_t = torch.zeros(batch_num, N_max).long()
        for b in range(batch_num):
            boxes_t[b, 0:boxes[b].shape[0]] = boxes[b]
            labels_t[b, 0:boxes[b].shape[0]] = labels[b]
        frame_label = torch.stack(frame_label)
        loc = torch.stack(loc)
        scale_t = torch.stack(scale)
        return img, boxes_t, labels_t, frame_label, loc, scale_t


    def load_image(self, idx, size, train=False):
        # nl = self.fnames[idx].split('/')
        # fn = os.path.join(r"D:\ubuntu\datasets\fetus_seg_data\Data_labeled", nl[-2], nl[-1])
        # img = Image.open(os.path.join(self.root, fn))
        img = Image.open(os.path.join(self.root, self.fnames[idx]))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        boxes = self.boxes[idx].clone()
        boxes[:, :2].clamp_(min=1)
        boxes[:, 2].clamp_(max=float(img.size[1]) - 1)
        boxes[:, 3].clamp_(max=float(img.size[0]) - 1)
        labels = self.labels[idx].clone()
        frame_label = self.frame_labels[idx].clone()
        if train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            # TODO: other augmentation (img, boxes)
            if self.augmentation is not None:
                img, boxes = self.augmentation(img, boxes)
            # standard procedure
            if random.random() < 0.5:
                img, boxes, loc, scale = random_resize_fix(img, boxes, size, self.img_scale_min)
            else:
                img, boxes, loc, scale = center_fix(img, boxes, size)
        else:
            img, boxes, loc, scale = center_fix(img, boxes, size)
        hw = boxes[:, 2:] - boxes[:, :2]  # [N,2]
        area = hw[:, 0] * hw[:, 1]  # [N]
        mask = area >= self.boxarea_th
        boxes = boxes[mask]
        labels = labels[mask]
        return img, boxes, labels, frame_label, loc, scale


    def load_mosaic2(self, index):
        img, boxes, labels, frame_label, loc, scale = self.load_mosaic(index)
        img2, boxes2, labels2, frame_label2, loc2, scale2 = self.load_mosaic(random.randint(0, len(self.fnames) - 1))

        r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
        img = (img * r + img2 * (1 - r)).astype(np.uint8)
        boxes = torch.cat((boxes, boxes2), dim=0)
        labels = torch.cat((labels, labels2), dim=0)
        frame_label = torch.cat((frame_label, frame_label2), dim=0)
        loc = torch.cat((loc, loc2), dim=0)
        scale = torch.cat((scale, scale2), dim=0)

        return img, boxes, labels, frame_label, loc, scale


    def load_mosaic(self, index):
        # loads images in a mosaic

        boxes4, labels4, frame_label4, loc4, scale4 = [], [], [], [], []
        s = self.size // 2
        yc, xc = s, s  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, boxes, labels, frame_label, loc, scale = self.load_image(index, s)
            w, h = img.size
            img = np.array(img)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((self.size, self.size, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            if boxes.size(0) > 0:
                boxes[:, 0::2] += padh
                boxes[:, 1::2] += padw

            boxes4.append(boxes)
            labels4.append(labels)
            frame_label4.append(frame_label.unsqueeze(0))
            loc4.append(loc.unsqueeze(0))
            scale4.append(scale)

        # Concat/clip labels
        boxes4 = torch.cat(boxes4, dim=0)
        boxes4.clamp_(0, 2 * s)  # use with random_perspective
        labels4 = torch.cat(labels4, dim=0).unsqueeze(1)
        frame_label4 = torch.cat(frame_label4, dim=0)
        loc4 = torch.cat(loc4, dim=0)

        scale4 = torch.tensor(scale4).unsqueeze(1)

        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, boxes4 = random_perspective(img4, boxes4.numpy())  # border to remove

        return img4, torch.from_numpy(boxes4), labels4, frame_label4, loc4, scale4



def random_perspective(img, boxes=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # ax[1].imshow(img[:, :, ::-1])  # warped
    # plt.show()

    # Transform label coordinates
    n = len(boxes)
    if n:
        boxes = boxes[:, [1, 0, 3, 2]]  # y1x1y2x2 -> x1y1x2y2

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        #i = box_candidates(box1=boxes.T * s, box2=xy.T)
        boxes = xy#[i]

        boxes = boxes[:, [1, 0, 3, 2]]  # x1y1x2y2 -> y1x1y2x2

    return img, boxes



def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates



def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT) 
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:,3]
        xmax = w - boxes[:,1]
        boxes[:,1] = xmin
        boxes[:,3] = xmax
    return img, boxes



def center_fix(img, boxes, size):
    w, h = img.size
    size_min = min(w,h)
    size_max = max(w,h)
    sw = sh = float(size) / size_max
    ow = int(w * sw + 0.5)
    oh = int(h * sh + 0.5)
    ofst_w = round((size - ow) / 2.0)
    ofst_h = round((size - oh) / 2.0)
    img = img.resize((ow,oh), Image.BILINEAR)
    img = img.crop((-ofst_w, -ofst_h, size-ofst_w, size-ofst_h))
    if boxes.shape[0] != 0:
        boxes = boxes*torch.FloatTensor([sh,sw,sh,sw])
        boxes += torch.FloatTensor([ofst_h, ofst_w, ofst_h, ofst_w])
    loc = torch.FloatTensor([ofst_h, ofst_w, ofst_h+oh, ofst_w+ow])
    return img, boxes, loc, sw



def random_resize_fix(img, boxes, size, img_scale_min):
    w, h = img.size
    size_min = min(w,h)
    size_max = max(w,h)
    scale_rate = float(size) / size_max
    scale_rate *= random.uniform(img_scale_min, 1.0)
    ow, oh = int(w * scale_rate + 0.5), int(h * scale_rate + 0.5)
    img = img.resize((ow,oh), Image.BILINEAR)
    if boxes.shape[0] != 0:
        boxes = boxes*torch.FloatTensor([scale_rate, scale_rate, scale_rate, scale_rate])
    max_ofst_h = size - oh
    max_ofst_w = size - ow
    ofst_h = random.randint(0, max_ofst_h)
    ofst_w = random.randint(0, max_ofst_w)
    img = img.crop((-ofst_w, -ofst_h, size-ofst_w, size-ofst_h))
    if boxes.shape[0] != 0:
        boxes += torch.FloatTensor([ofst_h, ofst_w, ofst_h, ofst_w])
    loc = torch.FloatTensor([ofst_h, ofst_w, ofst_h+oh, ofst_w+ow])
    return img, boxes, loc, scale_rate



COLOR_TABLE = [
    'Red', 'Green', 'Blue', 'Yellow',
    'Purple', 'Orange', 'DarkGreen', 'Purple',
    'YellowGreen', 'Maroon', 'Teal',
    'DarkGoldenrod', 'Peru', 'DarkRed', 'Tan',
    'AliceBlue', 'LightBlue', 'Cyan', 'Teal',
    'SpringGreen', 'SeaGreen', 'Lime', 'DarkGreen',
    'YellowGreen', 'Ivory', 'Olive', 'DarkGoldenrod',
    'Orange', 'Tan', 'Peru', 'Seashell',
    'Coral', 'RosyBrown', 'Maroon', 'DarkRed',
    'WhiteSmoke', 'LightGrey', 'Gray'
] * 10



def draw_bbox_text(drawObj, ymin, xmin, ymax, xmax, text, color, bd=2):
    drawObj.rectangle((xmin, ymin, xmax, ymin+bd), fill=color)
    drawObj.rectangle((xmin, ymax-bd, xmax, ymax), fill=color)
    drawObj.rectangle((xmin, ymin, xmin+bd, ymax), fill=color)
    drawObj.rectangle((xmax-bd, ymin, xmax, ymax), fill=color)
    strlen = len(text)
    drawObj.rectangle((xmin, ymin, xmin+strlen*6+5, ymin+12), fill=color)
    drawObj.text((xmin+3, ymin), text)



def show_bbox(img, boxes, labels, NAME_TAB, file_name=None, scores=None, 
                matplotlib=False, lb_g=True):
    '''
    img:      FloatTensor(3, H, W)
    boxes:    FloatTensor(N, 4)
    labels:   LongTensor(N)
    NAME_TAB: ['background', 'class_1', 'class_2', ...]
    file_name: 'out.bmp' or None
    scores:   FloatTensor(N)
    '''
    if lb_g: bg_idx = 0
    else: bg_idx = -1
    if not isinstance(img, Image.Image):
        img = transforms.ToPILImage()(img)
    drawObj = ImageDraw.Draw(img)
    for box_id in range(boxes.shape[0]):
        lb = int(labels[box_id])
        if lb > bg_idx:
            box = boxes[box_id]
            # if NAME_TAB is not None:
            if scores is None:
                draw_bbox_text(drawObj, box[0], box[1], box[2], box[3], NAME_TAB[lb], 
                    color=COLOR_TABLE[lb])
            else:
                str_score = str(float(scores[box_id]))[:5]
                str_out = NAME_TAB[lb] + ': ' + str_score
                draw_bbox_text(drawObj, box[0], box[1], box[2], box[3], str_out, 
                    color=COLOR_TABLE[lb])
    if file_name is not None:
        img.save(file_name)
    else:
        if matplotlib:
            plt.imshow(img, aspect='equal')
            plt.show()
        else: img.show()



if __name__ == '__main__':

    import augment
    def aug_func_demo(img, boxes):
        if random.random() < 0.9:
            img, boxes = augment.colorJitter(img, boxes, 
                            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        if random.random() < 0.9:
            img, boxes = augment.random_rotation(img, boxes, degree=5)
        if random.random() < 0.9:
            img, boxes = augment.random_crop_resize(img, boxes, size=512, 
                            crop_scale_min=0.2, aspect_ratio=[3./4, 4./3], remain_min=0.1, 
                            attempt_max=10)
        return img, boxes

    #TODO: parameters
    train = True
    size = 1025
    boxarea_th = 32
    img_scale_min = 0.8
    augmentation = None
    batch_size = 8
    csv_root  = 'D:\\dataset\\coco17\\images'
    csv_list  = '../data/coco_val2017.txt'
    csv_name  = '../data/coco_name.txt'

    
    dataset = Dataset_CSV(csv_root, csv_list, csv_name, 
        size=size, train=train, normalize=False, boxarea_th=boxarea_th,
        img_scale_min=img_scale_min, augmentation=augmentation)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
    for imgs, boxes, labels, locs, scales in dataloader:
        print(imgs.shape)
        print(boxes.shape)
        print(labels.shape)
        print(locs.shape)
        print(scales.shape)
        for i in range(len(boxes)):
            print(i, ': ', boxes[i].shape, labels[i].shape, locs[i], scales[i])
        # idx = int(input('idx:'))
        idx = 3
        print(labels[idx])
        print(boxes[idx][labels[idx]>0])
        print('avg px:', int(torch.min(locs[:, 2:] - locs[:, :2], dim=1)[0].mean()))
        show_bbox(imgs[idx], boxes[idx], labels[idx], dataset.LABEL_NAMES)
        # show_bbox(imgs[idx], boxes[idx], labels[idx], None)
        break
