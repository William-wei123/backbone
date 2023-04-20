# DarkNet系列

## DarkNet53
2018年
YOLO V3的backbone。Darknet53网络的具体结构如图所示，仿照resnet50所提出，其性能比resnet50优越。

![avatar](https://img-blog.csdnimg.cn/img_convert/851961d6eb2171a4c7e2fa73d90c0555.png#pic_center)

## CSPDarkNet
yoloV4的backbone为CSPDarkNet53，与Darknet53相比将[残差结构](https://github.com/William-wei123/backbone/blob/main/Darknet/DarkNet53.py#:~:text=(out)-,out%20%2B%3D%20residual,-return%20out)更改为[csp dense结构](https://github.com/William-wei123/backbone/blob/main/Darknet/cspdarknet53.py#:~:text=%E4%B8%AA%E5%8D%B7%E7%A7%AF-,%23%20%E6%AF%94%E4%BE%8B%E5%88%92%E5%88%86%20%E4%B8%941x1%E5%8D%B7,-split0%20%3D%20self),

![avatar](https://img-blog.csdnimg.cn/img_convert/b552f7033037bd2d7313a0e208d47028.png)
