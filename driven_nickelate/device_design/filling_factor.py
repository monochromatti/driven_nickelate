import matplotlib.image as mpimg

img = mpimg.imread("uc.png")

white = (img > 0.5).sum()
black = (img < 0.5).sum()

fraction = black / (black + white)
print(f"Gold filling fraction: {fraction * 100:.0f} %")

transmission_magnitude = 0.08 * fraction + (1 - fraction)
print(f"Transmission amplitude: {transmission_magnitude * 100:.0f} %")
