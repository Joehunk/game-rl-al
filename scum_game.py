import mss.tools
import mss.base
import mss.screenshot
import settings


def grab(mss: mss.base.MSSBase) -> mss.screenshot.ScreenShot:
    return mss.grab(settings.MONITOR)


def test_alignment_grab():
    with mss.mss() as sct:
        img = grab(sct)

        # Save to the picture file
        mss.tools.to_png(img.rgb, img.size, output="test_align.png")


if __name__ == "__main__":
    test_alignment_grab()
