import numpy as np
from monai.transforms.transform import MapTransform
from monai.transforms.utils import generate_spatial_bounding_box


def random_crop_from_box(img, seg, box, crop_size):
    """
    Safe random crop: automatically expands bounding box to avoid low >= high.
    box: (min_z, max_z, min_y, max_y, min_x, max_x)
    """
    D, H, W = img.shape[1:]  # CHWD
    cs = crop_size

    min_z, max_z, min_y, max_y, min_x, max_x = box

    # 1. Calculate box dimensions
    bz = max_z - min_z
    by = max_y - min_y
    bx = max_x - min_x

    # 2. Automatically expand box to at least crop_size
    if bz < cs[0]:
        extend = cs[0] - bz
        min_z = max(0, min_z - extend // 2)
        max_z = min(D, min_z + cs[0])

    if by < cs[1]:
        extend = cs[1] - by
        min_y = max(0, min_y - extend // 2)
        max_y = min(H, min_y + cs[1])

    if bx < cs[2]:
        extend = cs[2] - bx
        min_x = max(0, min_x - extend // 2)
        max_x = min(W, min_x + cs[2])

    # 3. Generate valid random starting point
    z1 = np.random.randint(min_z, max( min(max_z - cs[0], D - cs[0]), min_z ) + 1)
    y1 = np.random.randint(min_y, max( min(max_y - cs[1], H - cs[1]), min_y ) + 1)
    x1 = np.random.randint(min_x, max( min(max_x - cs[2], W - cs[2]), min_x ) + 1)

    z2, y2, x2 = z1 + cs[0], y1 + cs[1], x1 + cs[2]

    crop_img = img[:, z1:z2, y1:y2, x1:x2]
    crop_seg = seg[:, z1:z2, y1:y2, x1:x2]

    return crop_img, crop_seg


class SmallOrganCropd(MapTransform):
    """
    Fully compatible version:
    1) small organs take priority
    2) if small organs not present -> foreground fallback
    3) if foreground not present -> background crop
    """

    def __init__(
        self,
        keys,
        small_organs=(1, 2, 3),
        spatial_size=(96, 96, 96),
        num_samples=1,
    ):
        super().__init__(keys)
        self.small_organs = small_organs
        self.crop_size = spatial_size
        self.num_samples = num_samples

    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]
        seg = d[self.keys[1]]

        # Ensure img and seg are numpy arrays, handle MetaTensor if present
        if hasattr(img, 'as_tensor'):
            img = np.asarray(img)
        if hasattr(seg, 'as_tensor'):
            seg = np.asarray(seg)
        
        # Ensure numpy arrays
        img = np.asarray(img)
        seg = np.asarray(seg)

        samples = []

        for _ in range(self.num_samples):

            # 1. small organ mask
            # seg may be (C, D, H, W) or (D, H, W) format, need to handle
            if seg.ndim == 4:
                seg_3d = seg[0]  # take first channel
            else:
                seg_3d = seg
            
            small_mask = np.isin(seg_3d, self.small_organs).astype(np.uint8)
            if small_mask.sum() > 0:
                box_start, box_end = generate_spatial_bounding_box(small_mask)
                # Convert box_start and box_end to (min_z, max_z, min_y, max_y, min_x, max_x) format
                box = (box_start[0], box_end[0], box_start[1], box_end[1], box_start[2], box_end[2])
                crop_img, crop_seg = random_crop_from_box(
                    img, seg, box, self.crop_size
                )
                samples.append({self.keys[0]: crop_img, self.keys[1]: crop_seg})
                continue

            # 2. fallback: any foreground
            fg_mask = (seg_3d > 0).astype(np.uint8)
            if fg_mask.sum() > 0:
                box_start, box_end = generate_spatial_bounding_box(fg_mask)
                # Convert box_start and box_end to (min_z, max_z, min_y, max_y, min_x, max_x) format
                box = (box_start[0], box_end[0], box_start[1], box_end[1], box_start[2], box_end[2])
                crop_img, crop_seg = random_crop_from_box(
                    img, seg, box, self.crop_size
                )
                samples.append({self.keys[0]: crop_img, self.keys[1]: crop_seg})
                continue

            # 3. fallback: pure random background crop
            # img shape is (C, D, H, W) - CHWD format
            C, D, H, W = img.shape
            z1 = np.random.randint(0, max(1, D - self.crop_size[0] + 1))
            y1 = np.random.randint(0, max(1, H - self.crop_size[1] + 1))
            x1 = np.random.randint(0, max(1, W - self.crop_size[2] + 1))

            z2, y2, x2 = (
                z1 + self.crop_size[0],
                y1 + self.crop_size[1],
                x1 + self.crop_size[2],
            )
            slices = (slice(None), slice(z1, z2), slice(y1, y2), slice(x1, x2))

            crop_img, crop_seg = img[slices], seg[slices]
            samples.append({self.keys[0]: crop_img, self.keys[1]: crop_seg})

        # Return single dict when num_samples=1, otherwise return list (matching RandCropByPosNegLabeld behavior)
        if self.num_samples == 1:
            return samples[0]
        return samples