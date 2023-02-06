import typing as T
from pathlib import Path

import cv2 as cv

import numpy as np
import numpy.polynomial.polynomial as poly

from PlotDigitizerModule.grid import remove_grid
from PlotDigitizerModule.trajectory import find_trajectory, normalize
from PlotDigitizerModule.geometry import Point, find_origin

# NOTE: remember these are cv coordinates and not numpy.
locations_: T.List[Point] = []
points_: T.List[Point] = []

img_: np.ndarray = np.zeros((1, 1))
img_orig: np.ndarray = np.zeros((1, 1))
h_, w_ = 1, 1


def plot_traj(traj, outfile: Path):
    global locations_
    import matplotlib.pyplot as plt

    x, y = zip(*traj)
    plt.figure()
    plt.rcParams["figure.figsize"] = [2*(w_//h_), 2]
    plt.rcParams["figure.autolayout"] = True

    plt.subplot(211)
    plt.imshow(img_orig, interpolation="nearest", cmap="gray", aspect='auto')
    plt.axis(False)
    plt.title("Original")

    plt.subplot(212)
    plt.title("Reconstructed")
    plt.plot(x, y)
    plt.tight_layout()
    if not str(outfile):
        plt.show()
    else:
        plt.savefig(outfile)
    plt.close()


def list_to_points(points) -> T.List[Point]:
    ps = [Point.fromCSV(x) for x in points]
    return ps


def axis_transformation(p, P: T.List[Point]):
    """Compute m and offset for model Y = m X + offset that is used to transform
    axis X to Y"""

    # Currently only linear maps and only 2D.
    px, py = zip(*p)
    Px, Py = zip(*P)
    offX, sX = poly.polyfit(px, Px, 1)
    offY, sY = poly.polyfit(py, Py, 1)
    return ((sX, sY), (offX, offY))


def transform_axis(img, erase_near_axis: int = 0):
    global locations_
    global points_
    # extra: extra rows and cols to erase. Help in containing error near axis.
    # compute the transformation between old and new axis.
    T = axis_transformation(points_, locations_)
    p = find_origin(locations_)
    offCols, offRows = p.x, p.y
    img[:, : offCols + erase_near_axis] = params_["background"]
    img[-offRows - erase_near_axis :, :] = params_["background"]
    return T


def _find_trajectory_colors(
    img: np.ndarray, plot: bool = False
) -> T.Tuple[int, T.List[int]]:
    # Each trajectory color x is bounded in the range x-3 to x+2 (interval of
    # 5) -> total 51 bins. Also it is very unlikely that colors which are too
    # close to each other are part of different trajecotries. It is safe to
    # assme a binwidth of at least 10px.
    hs, bs = np.histogram(img.flatten(), 255 // 10, (0, img.max()))

    # Now a trajectory is only trajectory if number of pixels close to the
    # width of the image (we are using at least 75% of width).
    hs[hs < img.shape[1] * 3 // 4] = 0

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.bar(bs[:-1], np.log(hs))
        plt.xlabel("color")
        plt.ylabel("log(#pixel)")
        plt.show()

    # background is usually the color which is most count. We can find it
    # easily by sorting the histogram.
    hist = sorted(zip(hs, bs), reverse=True)

    # background is the most occuring pixel value.
    bgcolor = int(hist[0][1])

    # we assume that bgcolor is close to white.
    if bgcolor < 128:
        quit(-1)

    # If the background is white, search from the trajectories from the black.
    trajcolors = [int(b) for h, b in hist if h > 0 and b / bgcolor < 0.5]
    return bgcolor, trajcolors


def compute_foregrond_background_stats(img) -> T.Dict[str, float]:
    """Compute foreground and background color."""
    params: T.Dict[str, T.Any] = {}
    # Compute the histogram. It should be a multimodal histogram. Find peaks
    # and these are the colors of background and foregorunds. Currently
    # implementation is very simple.
    bgcolor, trajcolors = _find_trajectory_colors(img)
    params["background"] = bgcolor
    params["timeseries_colors"] = trajcolors
    return params


def process_image(img):
    global params_
    global args_
    params_ = compute_foregrond_background_stats(img)

    T = transform_axis(img, erase_near_axis=3)
    assert img.std() > 0.0, "No data in the image!"

    # extract the plot that has color which is farthest from the background.
    trajcolor = params_["timeseries_colors"][0]
    img = normalize(img)
    traj, img = find_trajectory(img, trajcolor, T)
    return traj


def run(img_path, plot=None, output=None):
    global locations_, points_
    global img_, img_orig
    global h_, w_

    infile = Path(img_path)
    assert infile.exists(), f"{infile} does not exists."

    # reads into gray-scale.
    img_orig = cv.imread(str(infile))
    img_ = cv.imread(str(infile), 0)
    (thresh, img_bw) = cv.threshold(img_, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    img_ = cv.bitwise_not(img_bw)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_op = cv.GaussianBlur(img_, (9, 9), 100)
    img_op = cv.dilate(img_op, kernel, iterations=5)
    img_op = cv.erode(img_op, kernel, iterations=5)
    (thresh, img_op) = cv.threshold(img_op, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    img_op = cv.bitwise_or(img_, img_op)
    img_ = cv.bitwise_not(cv.bitwise_xor(img_bw, cv.bitwise_not(img_op)))
    img_ = normalize(img_)

    (h_, w_) = img_.shape

    # remove grids.
    img_ = remove_grid(img_)

    # rescale it again.
    img_ = normalize(img_)
    assert img_.max() <= 255
    assert img_.min() < img_.mean() < img_.max(), "Could not read meaningful data"

    points_ = list_to_points(["0,0", "5,0", "0,2"])
    locations_ = list_to_points(["2,2", str(2 + w_//5)+",2", "2,"+str(2 + h_//2)])

    traj = process_image(img_)

    if plot is not None:
        plot_traj(traj, plot)

    outfile = output or f"{img_path}.traj.csv"
    with open(outfile, "w") as f:
        for r in traj:
            f.write("%g %g\n" % (r))

    return traj


def main():
    image_path = r"C:\Users\91886\Downloads\graph2.jpeg"
    plot_name = "output_.png"

    run(img_path=image_path, plot=plot_name)


if __name__ == "__main__":
    main()
