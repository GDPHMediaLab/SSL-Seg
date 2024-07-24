from re import L
import numpy as np
from monai import transforms
from torch import scalar_tensor, zero_

class ConvertToMultiChannelBasedOnBratsClassesd(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            result.append(d[key] == 2)
            d[key] = np.concatenate(result, axis=0).astype(np.float32)
        return d


def get_mae_pretrain_transforms(args):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                clip=True
            ),
            transforms.SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.CropForegroundd(keys=["image"], source_key="image",
                                       k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.num_samples,
                random_center=True,
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=0),
            transforms.RandFlipd(keys=["image"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=1),
            transforms.RandFlipd(keys=["image"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=2),
            transforms.ToTensord(keys=["image"]),
        ]
    )
    return train_transform



def get_simmim_pretrain_transforms(args):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                clip=True
            ),
            transforms.SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.CropForegroundd(keys=["image"], source_key="image",
                                       k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.num_samples,
                random_center=True,
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=0),
            transforms.RandFlipd(keys=["image"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=1),
            transforms.RandFlipd(keys=["image"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=2),
            transforms.ToTensord(keys=["image"]),
        ]
    )
    return train_transform



def get_mocov3_pretrain_transforms(args):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["fc", "zc"]),
            transforms.AddChanneld(keys=["fc", "zc"]),
            transforms.Orientationd(keys=["fc", "zc"], axcodes="RAS"),
            transforms.ScaleIntensityRanged(keys=["fc", "zc"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            transforms.CropForegroundd(keys=["fc", "zc"], source_key="fc"),
            transforms.RandSpatialCropSamplesd(
                keys=["fc","zc"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.num_samples,
                random_center=True,
                random_size=False,
            ),

            # 对 fc 应用的转换
            transforms.RandFlipd(keys=["fc"], prob=1.0, spatial_axis=0),
            transforms.RandRotate90d(keys=["fc"], prob=1.0, max_k=3),  # 增加旋转角度
            transforms.RandScaleIntensityd(keys=["fc"], factors=0.2, prob=1.0),  # 增加缩放强度
            transforms.RandShiftIntensityd(keys=["fc"], offsets=0.2, prob=1.0),
            transforms.RandGaussianNoised(keys=["fc"], prob=1.0, mean=0.0, std=0.1),  # 添加高斯噪声

            # 对 zc 应用的转换
            transforms.RandFlipd(keys=["zc"], prob=1.0, spatial_axis=1),
            transforms.Rand3DElasticd(keys=["zc"], prob=1.0, sigma_range=(5,7), magnitude_range=(50, 150),padding_mode='zeros',mode='nearest'),  # 应用弹性变形
            transforms.RandScaleIntensityd(keys=["zc"], factors=0.1, prob=1.0),
            transforms.RandAdjustContrastd(keys=["zc"], prob=1.0, gamma=(0.5, 1.5)),  # 调整对比度
            transforms.RandCoarseDropoutd(
                keys=["zc"],
                holes=6,
                spatial_size=(10, 10, 10),
                prob=0.5,
                max_holes=None
            ),
            transforms.ToTensord(keys=["fc", "zc"])
        ]
    )
    return train_transform



def get_simclr_pretrain_transforms(args):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["fc", "zc"]),
            transforms.AddChanneld(keys=["fc", "zc"]),
            transforms.Orientationd(keys=["fc", "zc"], axcodes="RAS"),
            transforms.ScaleIntensityRanged(keys=["fc", "zc"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            transforms.CropForegroundd(keys=["fc", "zc"], source_key="fc"),
            transforms.RandSpatialCropSamplesd(
                keys=["fc","zc"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.num_samples,
                random_center=True,
                random_size=False,
            ),

            # 对 fc 应用的转换
            transforms.RandFlipd(keys=["fc"], prob=1.0, spatial_axis=0),
            transforms.RandRotate90d(keys=["fc"], prob=1.0, max_k=3),  # 增加旋转角度
            transforms.RandScaleIntensityd(keys=["fc"], factors=0.2, prob=1.0),  # 增加缩放强度
            transforms.RandShiftIntensityd(keys=["fc"], offsets=0.2, prob=1.0),
            transforms.RandGaussianNoised(keys=["fc"], prob=1.0, mean=0.0, std=0.1),  # 添加高斯噪声

            # 对 zc 应用的转换
            transforms.RandFlipd(keys=["zc"], prob=1.0, spatial_axis=1),
            transforms.Rand3DElasticd(keys=["zc"], prob=1.0, sigma_range=(5,7), magnitude_range=(50, 150),padding_mode='zeros',mode='nearest'),  # 应用弹性变形
            transforms.RandScaleIntensityd(keys=["zc"], factors=0.1, prob=1.0),
            transforms.RandAdjustContrastd(keys=["zc"], prob=1.0, gamma=(0.5, 1.5)),  # 调整对比度
            transforms.RandCoarseDropoutd(
                keys=["zc"],
                holes=6,
                spatial_size=(10, 10, 10),
                prob=0.5,
                max_holes=None
            ),
            transforms.ToTensord(keys=["fc", "zc"])
        ]
    )
    return train_transform



def get_vis_transforms(args):
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"],
                                    axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z)
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    return val_transform


def get_raw_transforms(args):
    if args.dataset == 'btcv':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"],
                                        axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

class Resize():
    def __init__(self, scale_params):
        self.scale_params = scale_params

    def __call__(self, img):
        scale_params = self.scale_params
        shape = img.shape[1:]
        assert len(scale_params) == len(shape)
        spatial_size = []
        for scale, shape_dim in zip(scale_params, shape):
            spatial_size.append(int(scale * shape_dim))
        transform = transforms.Resize(spatial_size=spatial_size, mode='nearest')
        return transform(img)

def get_post_transforms(args):
    if args.dataset == 'btcv':
        if args.test:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
        else:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
    elif args.dataset == 'msd_brats':
        post_pred = transforms.Compose([transforms.EnsureType(), transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)])
        post_label = transforms.Identity()
    return post_pred, post_label