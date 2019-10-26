# core/yolov3-->decode(self, conv_output, anchors, stride) 疑难点解读

conv_output：经过darknet53以及yoloblock卷积层特征提取后得到的8x8x255、16x16x255、32x32x255特征图的中的任意一个

anchors：预设的三种尺寸anchor中的一种，每一种包含三种宽高比

stride：8、16、32中任意一个，即416x416输入图在输出时被缩放到的尺度

```
anchor_per_scale = len(anchors) 
这里的anchor_per_scale为3，即3种宽高比
```
```
conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))
conv_output原本是batch_sizex8x8x255(这里用8举例),本步操作将255拆成3部分，分别对应3个预选框, 每个维度85
```
```
conv_raw_dxdy = conv_output[:, :, :, :, 0:2] tx、ty
conv_raw_dwdh = conv_output[:, :, :, :, 2:4] tw、th
conv_raw_conf = conv_output[:, :, :, :, 4:5] 占一个位的置信度
conv_raw_prob = conv_output[:, :, :, :, 5: ] 占80个位的类别，注意，这里是进行80分类
```

上面都好理解，以下部分烧脑，算了好几遍才明白本段代码的作用，需要了解tf.tile, tf.newaxis方法的用处
```
y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
xy_grid = tf.cast(xy_grid, tf.float32)
```
接下来是这部分代码的过程图示，针对第1~3行
<div>
<img src="./images/yolov3_decode代码中生成特征图单元格坐标矩阵代码解读过程图.jpg">
<div>
