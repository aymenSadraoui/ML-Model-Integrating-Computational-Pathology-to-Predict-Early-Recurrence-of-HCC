import torch
from tqdm import tqdm
import numpy as np
import cv2
from utils.model_archi import IndepResNetModel

pej_color = np.array([220, 20, 60])  # crimson     #DC143C
non_pej_color = np.array([255, 215, 0])  # gold       #FFD700
healthy_color = np.array([50, 205, 50])  # lime green #32CD32
color2class = {
    tuple(healthy_color): "healthy",
    tuple(non_pej_color): "non pej",
    tuple(pej_color): "pej",
}

def gen_image_from_coords(coords_x, coords_y, y_har, step, colors):
    image = 255 + np.zeros(
        (int(coords_y.max()) + 2 * step, int(coords_x.max()) + 2 * step, 3),
        dtype=np.uint8,
    )
    for x, y, p in zip(coords_x, coords_y, y_har):
        image[int(y) : int(y) + step + 1, int(x) : int(x) + step + 1] = colors[p.item()]
    return image


def load_models(pth):
    models = []
    for i in tqdm(range(1, 6), desc="loading TripleResnet34"):
        model = IndepResNetModel().cuda()
        weights = torch.load(f"{pth}/TripleIndepResNet34_Fold{i}.pt", weights_only=False)
        model.load_state_dict(weights["model"])
        models.append(model)
    return models


def get_pred_proba_multi(model, data_loader):
    y_preds = torch.zeros(0, dtype=torch.long, device="cpu")
    y_true = torch.zeros(0, dtype=torch.long, device="cpu")
    y_proba = torch.zeros(0, dtype=torch.long, device="cpu")
    model.eval()
    with torch.no_grad():
        for inputs, target in tqdm(data_loader, desc='Prediction NT/NP/P'):
            im1, im2, im3 = inputs
            im1, im2, im3, target = (
                im1.cuda(),
                im2.cuda(),
                im3.cuda(),
                target.long().squeeze().cuda(),
            )
            proba = torch.softmax(model(im1, im2, im3), dim=1)
            pred = proba.argmax(1).detach()
            y_preds = torch.cat([y_preds, pred.view(-1).cpu()])
            y_true = torch.cat([y_true, target.view(-1).cpu()])
            y_proba = torch.cat([y_proba, proba.cpu()])
            del proba, inputs, target, im1, im2, im3, pred
    return y_true, y_preds, y_proba


def get_RdYlGr_masks(image):
    # Define the color range to search for
    lower_red, upper_red = pej_color - 10, pej_color + 10
    lower_yellow, upper_yellow = non_pej_color - 10, non_pej_color + 10
    lower_green, upper_green = healthy_color - 10, healthy_color + 10

    # Threshold the image using the color range
    mask_rd = cv2.inRange(image, lower_red, upper_red)
    mask_yl = cv2.inRange(image, lower_yellow, upper_yellow)
    mask_gr = cv2.inRange(image, lower_green, upper_green)

    masked_image_rd = cv2.bitwise_and(image, image, mask=mask_rd)
    masked_image_yl = cv2.bitwise_and(image, image, mask=mask_yl)
    masked_image_gr = cv2.bitwise_and(image, image, mask=mask_gr)
    return masked_image_rd, masked_image_yl, masked_image_gr


def get_largest_connected_area(masked_image, color):
    # Convert the image to grayscale
    gray = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # Find the contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if len(contours) != 0:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw the largest contour on a new image
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        # Apply the mask on the original image
        result = cv2.bitwise_and(masked_image, masked_image, mask=mask)
        num_pixels = cv2.countNonZero(cv2.inRange(result, color, color))
        area = cv2.contourArea(largest_contour)
    else:
        area = 0.0
        result = mask
    return result, area
