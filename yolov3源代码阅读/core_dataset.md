## DataSet作为一个类，生成用于训练的groundtrue的各类label

### 训练数据的数据结构定义
```
F:\project\keras-yolo\pic_word_bg_all/1_wordandzi/1560688055.0259538782_paper.jpg 296,844,345,893,0 566,120,609,163,0 71,855,106,890,0 304,681,384,761,0 418,75,450,107,0 282,528,315,561,0 512,884,576,948,0 275,947,316,988,0 291,77,362,148,0 579,280,661,362,0 47,367,87,407,0 524,243,560,279,0 417,192,452,227,0 531,789,568,826,0 361,233,399,271,0 358,573,408,623,0
```
以上是一条真实的训练数据，由两部分构成：图片路径+标注框集合；<br>
标注框集合中，以空格区分标注框，标注框数据由四部分组成：左上角x,左上角y,右下角x,右下角y,目标类别ID；<br>
因为这里训练的数据是软笔书法，仅存在一种类型，故目标类别ID保持为0；<br>

### 预选框尺寸
涉及到两个文件 data/anchors/coco_anchors.txt 和 data/anchors/basline_anchors.txt<br>
coco_anchors.txt内容：<br>
 `10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90,  156,198,  373,326 `<br>
basline_anchors.txt内容<br>
 `1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875 `<br>
两者的联系在如下：
```
10,13, 16,30, 33,23      除以 stride=8  --> 1.25,1.625, 2.0,3.75, 4.125,2.875              （检测小目标）
30,61, 62,45, 59,119     除以 stride=16 --> 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375      （检测中目标）
116,90, 156,198, 373,326 除以 stride=32 --> 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875   （检测大目标）
```
coco_anchors.txt中的预选框是针对图片的原始尺寸，而basline_anchors.txt中的预选框是针对yolov3网络输出的三种特征图的尺寸，我们之所以选用basline_anchors.txt中的数据作为anchors的尺寸，后面会讲到原因

### 类别文件
文件名：data/classes/coco.names，文件内容不展示，因为需要使用预训练模型，所以这里还是使用了原始的80分类，但将软笔书法作为ID=0的分类，覆盖原来的person分类；

### Dataset 构造函数（挑重点的说）
*  `self.train_input_sizes = cfg.TRAIN.INPUT_SIZE `，这里输入尺寸不止一个，训练过程中，每一轮中会随机挑选一个，如果训练的不多，很多尺寸就不会选择到，我们看下有哪些输入尺寸：`__C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608] `

*  `self.max_bbox_per_scale = 150 `，设置了每个缩放尺度stride下，每种宽高比最多可以有150个选择框

### Dataset 中的功能函数
* random_horizontal_flip
* random_crop
* random_translate<br>
这三个函数分别是随机镜像、随机裁剪、随机变换，代码在另外一个文件中已经说明，这里不再赘述

### Dataset/parse_annotation 初步从文件中解析训练数据中的标注框
重点在于image_preporcess这个函数，
* 输入为：
> `np.copy(image)` 原始图片的二进制数据 <br>
> `[self.train_input_size, self.train_input_size]` 向量 [ 随机挑选的输入尺寸，随机挑选的输入尺寸 ] ，比如 
```[416, 416]``` <br>
> `np.copy(bboxes)` 标注框数据列表，比如
```
[
 [296,844,345,893,0],
 [566,120,609,163,0],
 [71,855,106,890,0]
]
```

### utils/image_preporcess 图像预处理
```
def image_preporcess(image, target_size, gt_boxes=None):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    ih, iw    = target_size
    h,  w, _  = image.shape
    
    # 因为原始图像不总是正方形，强行缩放会失真，故针对其中单条边缩放至网络定义的输入尺寸，另一条边保持宽高比进行缩放，保证图像能够在输入尺寸中完全显示即可
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    
    # 对构造出的正方形输入区域进行颜色填充，并对其中的图像区域赋予图像像素值，空白区域保持填充色即可
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    
    # 将像素值归一化到0~1之间
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        # 此处重要，对选择框的左上角和右下角坐标进行缩放，缩放至网络定义的输入尺寸内
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        
        return image_paded, gt_boxes
```

### Dataset/preprocess_true_boxes 训练数据标注框处理至groundTrues，供实际训练使用
输入：经过上面所述的函数“parse_annotation”初步解析出来的bboxes，但这批bboxes的参数已经缩放至网络定义的输入尺寸范围内，且这是一张图片的检测框集合<br>
输出：label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes<br>
```
def preprocess_true_boxes(self, bboxes):

    global error_cnt

    label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                       5 + self.num_classes)) for i in range(3)]
    bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
    bbox_count = np.zeros((3,))

    for bbox in bboxes:
        bbox_coor = bbox[:4]
        bbox_class_ind = bbox[4]

        onehot = np.zeros(self.num_classes, dtype=np.float)
        onehot[bbox_class_ind] = 1.0
        uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
        deta = 0.01
        smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

        iou = []
        exist_positive = False
        for i in range(3):
            anchors_xywh = np.zeros((self.anchor_per_scale, 4))
            anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = self.anchors[i]

            iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
            iou.append(iou_scale)
            iou_mask = iou_scale > 0.3

            if np.any(iou_mask):
                xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                # haibingshuai:
                x_cc, y_cc = label[i].shape[:2]
                if xind >= x_cc:
                    error_cnt += 1
                    print('error_cnt increase to %d' % error_cnt)
                    xind = x_cc - 1
                if yind >= y_cc:
                    error_cnt += 1
                    print('error_cnt increase to %d' % error_cnt)
                    yind = y_cc - 1
                # ------------

                label[i][yind, xind, iou_mask, :] = 0
                label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                label[i][yind, xind, iou_mask, 4:5] = 1.0
                label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                bbox_count[i] += 1

                exist_positive = True

        if not exist_positive:
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = int(best_anchor_ind / self.anchor_per_scale)
            best_anchor = int(best_anchor_ind % self.anchor_per_scale)
            xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

            # haibingshuai:
            x_cc, y_cc = label[best_detect].shape[:2]
            if xind >= x_cc:
                error_cnt += 1
                print('error_cnt increase to %d' % error_cnt)
                xind = x_cc - 1
            if yind >= y_cc:
                error_cnt += 1
                print('error_cnt increase to %d' % error_cnt)
                yind = y_cc - 1
            # ------------

            label[best_detect][yind, xind, best_anchor, :] = 0
            label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
            label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
            label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

            bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
            bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
            bbox_count[best_detect] += 1
    label_sbbox, label_mbbox, label_lbbox = label
    sbboxes, mbboxes, lbboxes = bboxes_xywh
    return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
```
