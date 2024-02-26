import mss.tools
import mss.base
import mss.screenshot
import settings
import numpy as np
from torchvision.transforms import functional as F


def grab(mss: mss.base.MSSBase) -> mss.screenshot.ScreenShot:
    return mss.grab(settings.MONITOR)


def test_alignment_grab():
    with mss.mss() as sct:
        img = grab(sct)

        # Save to the picture file
        mss.tools.to_png(img.rgb, img.size, output="test_align.png")

def grab_for_ann_raw(mss: mss.base.MSSBase):
    screenshot = grab(mss)
    # Convert the screenshot to a numpy array
    # The MSS screenshot is in BGRA format, so we convert it to RGBA
    img_np = np.array(screenshot)
    img_np = np.flip(img_np[:, :, :3], axis=-1)  # Convert BGRA to RGB

    # Convert the numpy array to a PyTorch tensor
    # torchvision expects the tensor to be in C, H, W format
    img_tensor = F.to_tensor(img_np.copy())
    return img_tensor

if __name__ == "__main__":
    test_alignment_grab()
