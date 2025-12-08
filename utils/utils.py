import cv2
import pickle
import openslide
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def interpolate(size, nominateur, denominateur):
    """
    size = 1152 or 1024 or 512
    nominateur = 3DHisto resolution = 0.25
    denominateur = hamamatsu resolution = 0.46
    """
    return int((size * nominateur) // denominateur)


# Patch generation steps
def get_BrightandDark_perc(pil_image, bright_threshold=200, dark_threshold=20):
    grayscale_image = np.array(pil_image.convert("L"))
    # Count bright/white and dark/black pixels
    bright_pixels = np.sum(grayscale_image > bright_threshold)
    dark_pixels = np.sum(grayscale_image < dark_threshold)
    # Calculate percentages
    total_pixels = grayscale_image.size
    bright_percentage = bright_pixels / total_pixels
    dark_percentage = dark_pixels / total_pixels
    return bright_percentage, dark_percentage


def detect_tissue_regions(array_slide):
    gray = cv2.cvtColor(array_slide, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )  # Threshold the image to create a binary mask
    output = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # Find contours in the binary mask
    contours = output[0] if len(output) == 2 else output[1]
    largest_contour = max(
        contours, key=cv2.contourArea
    )  # Find the contour with the largest area (largest tissue region)
    x, y, w, h = cv2.boundingRect(
        largest_contour
    )  # Get the coordinates of the bounding rectangle around the largest tissue region
    return np.array([x, y, w, h])


def image_stats(image):
    """Compute the mean and standard deviation of each channel in the image."""
    mean, std = cv2.meanStdDev(image)
    mean = mean.flatten()
    std = std.flatten()
    return mean, std


def color_transfer(source, target):
    """Perform color transfer from the target image to the source image."""
    # Convert images from BGR to Lab color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2Lab)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2Lab)

    # Compute the mean and standard deviation of each channel
    mean_src, std_src = image_stats(source_lab)
    mean_tar, std_tar = image_stats(target_lab)

    # Subtract the mean from the source image
    source_lab = (source_lab - mean_src) / std_src

    # Scale by the standard deviation of the target and add the target mean
    source_lab = source_lab * std_tar + mean_tar

    # Clip the values to the valid range [0, 255] and convert back to uint8
    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)

    # Convert back to BGR color space
    transfer = cv2.cvtColor(source_lab, cv2.COLOR_Lab2RGB)

    return transfer


def generate_patches_from_wsi(
    slide_name,
    path_to_wsi,
    patch_size,
    step,
    path_to_patches,
    vis_scale,
    overview_path,
    coords_path,
    perc_bpx=0.05,
    perc_wpx=0.85,
    enlarge=20,
):
    real_enlarge = int(enlarge / vis_scale)
    slide_name = f"{path_to_wsi}/{slide_name}"
    slide = openslide.OpenSlide(f"{slide_name}")
    print(slide.dimensions)
    slide_w, slide_h = slide.dimensions
    W, H = int(slide_w * vis_scale), int(slide_h * vis_scale)
    array_slide = np.array(slide.get_thumbnail((W, H)).convert("RGB"))
    xywh = detect_tissue_regions(array_slide)
    real_x, real_y, real_w, real_h = np.array(xywh // vis_scale, np.int64)
    x, y, width, height = xywh
    x_start, y_start, x_end, y_end = (
        max(0, x - enlarge),
        max(0, y - enlarge),
        x + enlarge,
        y + enlarge,
    )
    real_x, real_y, real_w, real_h = (
        max(0, real_x - real_enlarge),
        max(0, real_y - real_enlarge),
        real_w + real_enlarge,
        real_h + real_enlarge,
    )
    scaled_slide = np.array(array_slide)[
        y_start : y_end + height, x_start : x_end + width
    ]

    slide_name = slide_name.split("/")[-1].split(".")[0]
    coords_x, coords_y = [], []
    range_x = range(real_x, real_x + real_w, step)
    range_y = range(real_y, real_y + real_h, step)
    pth1 = path_to_patches + slide_name.split(".")[0]
    Path(pth1).mkdir(exist_ok=True)
    with tqdm(total=len(range_x) * len(range_y), desc="Patch extraction") as pbar:
        for x in range_x:
            for y in range_y:
                patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert(
                    "RGB"
                )
                wpx, bpx = get_BrightandDark_perc(patch)
                if wpx < perc_wpx and bpx < perc_bpx:
                    coords_x.append(x)
                    coords_y.append(y)
                    patch.save(f"{pth1}/patch_x_{str(x)}_y_{str(y)}.jpg")
                    pbar.update(1)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    fig.suptitle(slide_name, fontsize=18)
    ax1.imshow(array_slide)
    ax1.add_patch(Rectangle(xy=xywh[:2], width=xywh[-2], height=xywh[-1], edgecolor="k", lw=1.5, facecolor="none"))  # type: ignore
    ax2.imshow(scaled_slide)
    ax3.scatter(
        np.array(coords_x) * vis_scale - x_start,
        np.array(coords_y) * vis_scale - y_start,
        c="g",
        marker="s",
        s=8,
        alpha=0.3,
        edgecolor="yellow",
    )
    ax3.imshow(scaled_slide)
    for ax in [ax1, ax2, ax3]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{overview_path}/{slide_name}_overview.png", dpi=150)
    plt.show()

    to_save = {
        "coords_x": coords_x,
        "coords_y": coords_y,
        "xy_start_end": [x_start, y_start, x_end, y_end],
        "xywh_real": [real_x, real_y, real_w, real_h],
        "scaled_slide": scaled_slide,
        "vis_scale": vis_scale,
    }

    with open(f"{coords_path}/{slide_name}_coords_checkpoint.pickle", "wb") as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(slide_name, "done!")
