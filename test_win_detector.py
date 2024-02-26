import pytorch_device
import win_and_angle_detector_cnn
import torch
import scum_game
import torchvision.io
from torchvision import transforms
import mss

model = win_and_angle_detector_cnn.WinAndAngleDetector().to(
    pytorch_device.device, dtype=torch.float32
)
model.load_state_dict(torch.load("bar_goodboth.pth"))
model.eval()


def analyze_frame(mss):
    image = scum_game.grab_for_ann_raw(mss)
    image = transforms.Resize((400, 400))(image)
    image = image.unsqueeze(0)
    image = image.to(pytorch_device.device, dtype=torch.float32)

    class_output, qual_output = model(image)

    class_output = class_output.squeeze(0)
    qual_output = qual_output.squeeze(0)

    print(f"It is {class_output.item():.3f} {qual_output.item():.3f}")

    if class_output.item() > 0.8:
        return True
    return False


def analyze_file(filename):
    image = torchvision.io.read_image(filename).to(pytorch_device.device, dtype=torch.float32)
    image = transforms.Resize((400, 400))(image)
    image = image.unsqueeze(0)

    class_output, qual_output = model(image)
    class_output = class_output.squeeze(0)
    qual_output = qual_output.squeeze(0)

    print(f"It is {class_output.item():.3f} {qual_output.item():.3f}")

if __name__ == "__main__":
    # with mss.mss() as sct:
    #     while True:
    #         if analyze_frame(sct):
    #             break
    analyze_file('test_align.png')
