```
import numpy as np

bbox_coor = np.array([8.0, 8.0, 16.0, 16.0])
# bbox_xywh = [8, 8, 8, 8]
strides = np.array([8, 16, 32])
anchors = np.array([
    [[10, 13], [16, 30], [33, 23]],
    [[30, 61], [62, 45], [59, 119]],
    [[116, 90], [156, 198], [373, 326]],
])
print(np.shape(anchors))

bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
print('\nbbox_xywh:')
print(bbox_xywh)

bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

print('\nbbox_xywh_scaled:')
print(bbox_xywh_scaled)

i = 0
anchors_xywh = np.zeros((3, 4))
anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
anchors_xywh[:, 2:4] = anchors[i]
print('\nanchors_xywh:')
print(anchors_xywh)


arg1 = bbox_xywh_scaled[i][np.newaxis, :]
print('\narg1:')
print(arg1)

l = np.array([1, 2, -1])
print(l > 1)

print(111)
```
