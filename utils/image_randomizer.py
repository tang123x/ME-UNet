import numpy as np
import random
from scipy.interpolate import interp1d


def bezier_curve(points, num=256):
    t = np.linspace(0, 1, num)
    p0, p1, p2, p3 = points
    curve = (1 - t) ** 3 * p0 + \
            3 * (1 - t) ** 2 * t * p1 + \
            3 * (1 - t) * t ** 2 * p2 + \
            t ** 3 * p3
    return curve


def generate_bezier_mapping():
    p0 = 0.0
    p3 = 1.0
    p1 = random.uniform(0.2, 0.5)
    p2 = random.uniform(0.5, 0.8)
    curve = bezier_curve([p0, p1, p2, p3])
    x = np.linspace(0, 1, len(curve))
    return interp1d(x, curve, kind='linear', fill_value="extrapolate")


def apply_gcm(image):
    """
    Global Contrast Modulation (GCM).
    Adjusts overall tone distribution to simulate different scanner conditions.
    
    Args:
        image: Input image
        
    Returns:
        Globally adjusted image
    """
    img_min = image.min()
    img_max = image.max()
    norm_img = (image - img_min) / (img_max - img_min + 1e-8)
    mapped = generate_bezier_mapping()(norm_img.clip(0, 1))
    return mapped * (img_max - img_min) + img_min


def apply_osp(image, label, target_label, noise_std=0.05):
    """
    Organ-Selective Perturbation (OSP).
    Enhances robustness to local organ-wise intensity changes by introducing
    controlled multiplicative noise exclusively within a selected organ region.
    
    Args:
        image: Input image
        label: Label image
        target_label: Target organ label to perturb
        noise_std: Standard deviation of Gaussian noise (default: 0.05)
        
    Returns:
        Perturbed image with noise applied to target organ region
    """
    mask = (label == target_label).astype(np.float32)
    if mask.sum() == 0:
        return image
    noise = np.random.normal(loc=1.0, scale=noise_std, size=image.shape)
    return image * (1 - mask) + image * noise * mask


def apply_sbf(image):
    """
    Saliency-Balanced Fusion (SBF).
    Refines the augmented representation by adaptively balancing high-frequency
    and low-frequency components based on saliency information derived from
    image gradient. Enhances sensitivity to subtle structural boundaries while
    suppressing redundant intensity variations in homogeneous regions.
    
    Args:
        image: Input image (D, H, W) or (1, D, H, W)
        
    Returns:
        Saliency-balanced image
    """
    if image.ndim == 4 and image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
        
    grad = np.gradient(image)
    grad_mag = np.sqrt(sum([g ** 2 for g in grad]))
    
    smooth_factor = np.exp(-grad_mag / (grad_mag.mean() + 1e-5))
    alpha = 1.0 - smooth_factor
    fused_image = image * alpha + image.mean() * (1 - alpha)

    return fused_image


def apply_moa(
    image,
    label,
    fixed_labels=None,
    candidate_labels=None,
    noise_std=0.05
):
    """
    Multi-Level Organ-Aware Augmentation (MOA).
    A hierarchical augmentation strategy that modulates intensity and structure
    at three levels: GCM, OSP, and SBF.
    
    Args:
        image: Input image (D, H, W) or (1, D, H, W)
        label: Label image (D, H, W) or (1, D, H, W)
        fixed_labels: List of organ labels that must be augmented
        candidate_labels: List of optional organ labels for random augmentation
        noise_std: Standard deviation for OSP noise
        
    Returns:
        List of augmented image versions
    """
    image = image.astype(np.float32)
    label = label.astype(np.int32)
    unique_labels = np.unique(label)

    has_only_background = len(unique_labels) == 1 and unique_labels[0] == 0

    if fixed_labels is not None:
        fixed_labels = [f for f in fixed_labels if f in unique_labels]

    if candidate_labels is not None:
        candidate_labels = [c for c in candidate_labels if c in unique_labels]
        if fixed_labels:
            candidate_labels = [c for c in candidate_labels if c not in fixed_labels]
    else:
        candidate_labels = []

    gcm = apply_gcm(image).astype(np.float32)
    versions = [image, gcm]

    if has_only_background:
        gcm_variant = apply_gcm(image).astype(np.float32)
        versions.append(gcm_variant)
    else:
        if fixed_labels:
            for fixed in fixed_labels:
                lla_fixed = apply_osp(gcm, label, target_label=fixed, noise_std=noise_std).astype(np.float32)
                versions.append(lla_fixed)

        if candidate_labels:
            if candidate_labels:
                rand_label = random.choice(candidate_labels)
                lla_rand = apply_osp(gcm, label, target_label=rand_label, noise_std=noise_std).astype(np.float32)
                versions.append(lla_rand)

    last_ver = versions[-1]
    restore_batch = False
    if last_ver.ndim == 4 and last_ver.shape[0] == 1:
        last_ver = np.squeeze(last_ver, axis=0)
        restore_batch = True
    sbf = apply_sbf(last_ver).astype(np.float32)
    if restore_batch:
        sbf = np.expand_dims(sbf, axis=0)
    versions.append(sbf)

    return versions