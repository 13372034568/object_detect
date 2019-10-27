## 图像的一些处理技巧

### 水平翻转

def random_horizontal_flip(self, image, bboxes):

    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

    return image, bboxes
