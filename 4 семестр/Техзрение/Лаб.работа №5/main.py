import cv2
import numpy as np
import skimage

import matplotlib.pyplot as plt


def min_max_len_lines(lines):
    min_len = float('inf')
    max_len = 0
    min_len_line = None
    max_len_line = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            line_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if line_len > max_len:
                max_len = line_len
                max_len_line = line
            if line_len < min_len:
                min_len = line_len
                min_len_line = line

    return min_len, min_len_line, max_len, max_len_line


def min_max_radius_circles(circles):
    min_radius = float('inf')
    max_radius = 0
    min_circle = None
    max_circle = None

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if i[2] < min_radius:
                min_radius = i[2]
                min_circle = i
            if i[2] > max_radius:
                max_radius = i[2]
                max_circle = i

    return min_radius, min_circle, max_radius, max_circle


def detect_lines(src, rho, theta, threshold, minLineLength, maxLineGap):
    lines = cv2.HoughLinesP(
        image=src,
        rho=rho,
        theta=theta,
        threshold=threshold,
        lines=None,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )
    return lines


def draw_lines(src, lines, thickness=1, color=(0, 0, 255),
               marker_radius=2, dot_color1=(0, 255, 0), dot_color2=(0, 255, 0)):
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            x_1, y_1, x_2, y_2 = l[0], l[1], l[2], l[3]
            cv2.line(src, (x_1, y_1), (x_2, y_2), color, thickness, cv2.LINE_AA)
            cv2.circle(src, (x_1, y_1), marker_radius, dot_color1, -1)
            cv2.circle(src, (x_2, y_2), marker_radius, dot_color2, -1)

    return src


def parameters_space_lines(src, arr_ang_step=0.1, brightness=3):
    dots = int(round(360 / arr_ang_step))
    angles = np.linspace(-np.pi / 2, np.pi / 2, dots, endpoint=False)
    H, theta, rho = skimage.transform.hough_line(src, theta=angles)

    ang_step = 0.5 * np.diff(theta).mean()
    dis_step = 0.5 * np.diff(rho).mean()

    bounds = [np.rad2deg(theta[0] - ang_step), np.rad2deg(theta[-1] + ang_step), rho[-1] + dis_step, rho[0] - dis_step]
    parameters_space = cv2.cvtColor(np.float32(brightness * H / np.max(H)), cv2.COLOR_GRAY2RGB)
    return parameters_space, bounds


def detect_circles(src, dp, min_dis, par1, par2, min_r, max_r):
    circles = cv2.HoughCircles(
        image=src,
        method=cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dis,
        param1=par1,
        param2=par2,
        minRadius=min_r,
        maxRadius=max_r
    )
    return circles


def draw_circles(src, circs, color=(0, 0, 255), thickness=2):
    if circs is not None:
        circles = np.uint16(np.around(circs))
        for i in circles[0, :]:
            cv2.circle(src, (i[0], i[1]), i[2], color, thickness)
    return src


if __name__ == '__main__':
    # static
    render_to = 'renders'
    src_dir = 'source'
    curr_img = 'ci3.png'
    filename = f'{src_dir}/{curr_img}'
    clr_src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)
    src = cv2.cvtColor(clr_src, cv2.COLOR_BGR2GRAY)

    render_lines = False
    render_circles = True
    render_no_canny = True
    render_canny = True

    if render_lines:
        # changeable
        rho = 1
        theta = np.pi / 180
        threshold = 130
        threshold1 = 350
        threshold2 = 230
        apertureSize = 3
        minLineLength = 20
        maxLineGap = 5

        # no canny
        if render_no_canny:
            lines_src = detect_lines(src, rho, theta, threshold, minLineLength, maxLineGap)
            output_src = draw_lines(clr_src.copy(), lines_src)

            min_len, min_len_l, max_len, max_len_l = min_max_len_lines(lines_src)
            lines_count = len(lines_src)

            cv2.imwrite(f'{render_to}/hl_{curr_img}', output_src)
            print(f'max_len_line={max_len}p\nmin_len_line={min_len}p\nlines_count={lines_count}\n')

        # canny
        if render_canny:
            edge_src = cv2.Canny(src, threshold1, threshold2, None, apertureSize)
            lines_edge_src = detect_lines(edge_src, rho, theta, threshold, minLineLength, maxLineGap)
            output_edge_src = draw_lines(clr_src.copy(), lines_edge_src)
            ps_pace, bds = parameters_space_lines(edge_src)

            edge_min_len, edge_min_len_l, edge_max_len, edge_max_len_l = min_max_len_lines(lines_edge_src)
            edge_lines_count = len(lines_edge_src)

            cv2.imwrite(f'{render_to}/canny_{curr_img}', edge_src)
            plt.imshow(ps_pace, extent=bds, aspect=0.1)
            plt.savefig(f'{render_to}/canny_par_space_{curr_img}')
            cv2.imwrite(f'{render_to}/canny_hl_{curr_img}', output_edge_src)
            print(f'canny_max_len_line={edge_max_len}p\ncanny_min_len_line={edge_min_len}p\n'
                  f'canny_lines_count={edge_lines_count}')

    if render_circles:
        sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=5)
        edge_src = np.hypot(sobelx, sobely)
        edge_src = np.uint8(edge_src / np.max(edge_src) * 255)

        # changeable
        dp = 1
        min_dis = 10
        par1 = 40
        par2 = 10
        min_r = 30
        max_r = 95
        single_r = 90

        # no sobel (called canny. whtvr)
        if render_no_canny:
            circles_src = detect_circles(src, dp, min_dis, par1, par2, min_r, max_r)
            output_src = draw_circles(clr_src.copy(), circles_src)

            min_radius, min_circle, max_radius, max_circle = min_max_radius_circles(circles_src)
            circles_count = len(circles_src[0])

            circles_src2 = detect_circles(src, dp, min_dis, par1, par2, single_r, single_r)
            output_src2 = draw_circles(clr_src.copy(), circles_src2)

            cv2.imwrite(f'{render_to}/hc_{curr_img}', output_src)
            cv2.imwrite(f'{render_to}/hc_r={single_r}_{curr_img}', output_src2)
            print(f'min_radius={min_radius}p\nmax_radius={max_radius}p\ncircles_count={circles_count}\n')

        # sobel
        if render_canny:
            circles_edge_src = detect_circles(edge_src, dp, min_dis, par1, par2, min_r, max_r)
            output_edge_src = draw_circles(clr_src.copy(), circles_edge_src)

            edge_min_radius, edge_min_circle, edge_max_radius, edge_max_circle = min_max_radius_circles(circles_edge_src)
            edge_circs_count = len(circles_edge_src[0])

            circles_edge_src2 = detect_circles(edge_src, dp, min_dis, par1, par2, single_r, single_r)
            output_edge_src2 = draw_circles(clr_src.copy(), circles_edge_src2)

            cv2.imwrite(f'{render_to}/canny_{curr_img}', edge_src)
            cv2.imwrite(f'{render_to}/canny_hc_{curr_img}', output_edge_src)
            cv2.imwrite(f'{render_to}/canny_hc_r={single_r}_{curr_img}', output_edge_src2)
            print(f'min_radius={edge_min_radius}p\nmax_radius={edge_max_radius}p\ncircles_count={edge_circs_count}\n')
