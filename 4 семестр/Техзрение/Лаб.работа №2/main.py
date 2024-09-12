import numpy as np
import cv2


def shift_img(img, delta_x, delta_y):
    rows, cols = img.shape[0:2]
    t = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    return cv2.warpAffine(img, t, (cols, rows))


def reflect_img(img, flag: bool):
    rows, cols = img.shape[0:2]
    if (flag):
        t = np.float32([[1, 0, 0], [0, -1, rows - 1]])
    else:
        t = np.float32([[-1, 0, rows - 1], [0, 1, 0]])
    return cv2.warpAffine(img, t, (cols, rows))


def scale_img(img, scale_x, scale_y):
    rows, cols = img.shape[0:2]
    t = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
    return cv2.warpAffine(img, t, (int(cols * scale_x), int(rows * scale_y)))


def rotate_img(img, angle):
    rows, cols = img.shape[0:2]
    phi = np.radians(angle)
    t = np.float32([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0]])
    return cv2.warpAffine(img, t, (cols, rows))


def rotate_img_center(img, angle):
    rows, cols = img.shape[0:2]
    phi = np.radians(angle)
    t1 = np.float32([[1, 0, -(cols - 1) / 2], [0, 1, -(rows - 1) / 2], [0, 0, 1]])
    t2 = np.float32([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    t3 = np.float32([[1, 0, (cols - 1) / 2], [0, 1, (rows - 1) / 2], [0, 0, 1]])
    t = np.matmul(t3, np.matmul(t2, t1))[0:2, :]
    return cv2.warpAffine(img, t, (cols, rows))


def bevel_img(img, arg):
    rows, cols = img.shape[0:2]
    t = np.float32([[1, arg, 0], [0, 1, 0]])
    return cv2.warpAffine(img, t, (cols, rows))


def piecewise_linear_mapping(img, arg):
    rows, cols = img.shape[0:2]
    t = np.float32([[arg, 0, 0], [0, 1, 0]])
    img[:, int(cols / 2):, :] = cv2.warpAffine(img[:, int(cols / 2):, :], t, (cols - int(cols / 2), rows))
    return img


def projection(img, arr):
    rows, cols = img.shape[0:2]
    t = np.float32(arr)
    return cv2.warpPerspective(img, t, (cols, rows))


def polynomial(img, t):
    rows, cols = img.shape[0:2]
    t = np.array(t)
    I_polynomial = np.zeros(img.shape, img.dtype)

    x, y = np.meshgrid(np.arange(cols),

                       np.arange(rows))

    xnew = np.round(t[0, 0] + x * t[1, 0] +
                    y * t[2, 0] + x * x * t[3, 0] +
                    x * y * t[4, 0] +
                    y * y * t[5, 0]).astype(np.float32)
    ynew = np.round(t[0, 1] + x * t[1, 1] +
                    y * t[2, 1] + x * x * t[3, 1] +
                    x * y * t[4, 1] +
                    y * y * t[5, 1]).astype(np.float32)
    mask = np.logical_and(
        np.logical_and(xnew >= 0, xnew < cols),
        np.logical_and(ynew >= 0, ynew < rows))
    if img.ndim == 2:

        I_polynomial[ynew[mask].astype(int),

        xnew[mask].astype(int)] = \
            img[y[mask], x[mask]]
    else:
        I_polynomial[ynew[mask].astype(int),
        xnew[mask].astype(int), :] = \
            img[y[mask], x[mask], :]
    return I_polynomial


def sinusoidal(img):
    rows, cols = img.shape[0:2]
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    u = u + 20 * np.sin(2 * np.pi * v / 90)
    img_sinusoid = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
    return img_sinusoid


def barrel_distorsion(img):
    rows, cols = img.shape[0:2]
    xi, yi = np.meshgrid(np.arange(cols),
                         np.arange(rows))
    xmid = cols / 2.0
    ymid = rows / 2.0
    xi = xi - xmid
    yi = yi - ymid
    r, theta = cv2.cartToPolar(xi / xmid, yi / ymid)
    F3 = 0.1
    F5 = 0.12
    r = r + F3 * r ** 3 + F5 * r ** 5
    u, v = cv2.polarToCart(r, theta)
    u = u * xmid + xmid
    v = v * ymid + ymid
    I_barrel = \
        cv2.remap(img, u.astype(np.float32),
                  v.astype(np.float32), cv2.INTER_LINEAR)
    return I_barrel


def pincushion_distorsion(img):
    rows, cols = img.shape[0:2]
    xi, yi = np.meshgrid(np.arange(cols),
                         np.arange(rows))
    xmid = cols / 2.0
    ymid = rows / 2.0
    xi = xi - xmid
    yi = yi - ymid
    r, theta = cv2.cartToPolar(xi / xmid, yi / ymid)
    F3 = -0.1
    F5 = -0.12
    r = r + F3 * r ** 3 + F5 * r ** 5
    u, v = cv2.polarToCart(r, theta)
    u = u * xmid + xmid
    v = v * ymid + ymid
    I_barrel = \
        cv2.remap(img, u.astype(np.float32),
                  v.astype(np.float32), cv2.INTER_LINEAR)
    return I_barrel


def undistort_barrel(img):
    rows, cols = img.shape[0:2]
    xi, yi = np.meshgrid(np.arange(cols), np.arange(rows))
    xmid = cols / 2.0
    ymid = rows / 2.0
    xi = xi - xmid
    yi = yi - ymid
    r, theta = cv2.cartToPolar(xi / xmid, yi / ymid)

    F3 = 0.1
    F5 = 0.06

    r = r - F3 * r ** 3 - F5 * r ** 5
    u, v = cv2.polarToCart(r, theta)
    u = u * xmid + xmid
    v = v * ymid + ymid

    I_undistorted = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return I_undistorted


def undistort_pincushion(img):
    rows, cols = img.shape[0:2]
    xi, yi = np.meshgrid(np.arange(cols),
                         np.arange(rows))
    xmid = cols / 2.0
    ymid = rows / 2.0
    xi = xi - xmid
    yi = yi - ymid
    r, theta = cv2.cartToPolar(xi / xmid, yi / ymid)
    F3 = 0.5
    F5 = 1
    r = r + F3 * r ** 3 + F5 * r ** 5
    u, v = cv2.polarToCart(r, theta)
    u = u * xmid + xmid
    v = v * ymid + ymid
    I_barrel = \
        cv2.remap(img, u.astype(np.float32),
                  v.astype(np.float32), cv2.INTER_LINEAR)
    return I_barrel


def join_img(topPart, botPart):
    templ_size = 10
    templ = topPart[- templ_size:, :, :]
    res = cv2.matchTemplate(botPart, templ,
    cv2.TM_CCOEFF )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    result_img = np.zeros(( topPart . shape [0] + botPart . shape [0] - max_loc [1]
                            - templ_size , topPart . shape [1] , topPart . shape [2]) , dtype = np . uint8 )
    result_img[0: topPart.shape[0], :, :] = topPart
    result_img[topPart.shape[0]:, :, :] = botPart [ max_loc [1] + templ_size : , : , :]
    return result_img


def auto_join_img(imgs:list):
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, I_stitch = stitcher.stitch(imgs)
    return status, I_stitch

if __name__ == "__main__":
    path = 'tech-vision/source/img1.jpg'
    render_dir = 'tech-vision/renders/'

    source_img = cv2.imread(path)
    cv2.imwrite(f'{render_dir}cats.png', source_img)

    shifted_img = shift_img(source_img, delta_x=50, delta_y=100)
    cv2.imwrite(f'{render_dir}shifted_img.png', shifted_img)

    reflected_img_ox = reflect_img(source_img, flag=True)
    reflected_img_oy= reflect_img(source_img, flag=False)
    cv2.imwrite(f'{render_dir}reflect_img_ox.png', reflected_img_ox)
    cv2.imwrite(f'{render_dir}reflect_img_oy.png', reflected_img_oy)

    scaled_img_bigger = scale_img(source_img, 2, 2)
    scaled_img_smaller = scale_img(source_img, 0.5, 0.5)
    scaled_img_wtf = scale_img(source_img, 1, 0.5)
    cv2.imwrite(f'{render_dir}scaled_img_bigger.png', scaled_img_bigger)
    cv2.imwrite(f'{render_dir}scaled_img_smaller.png', scaled_img_smaller)
    cv2.imwrite(f'{render_dir}scaled_img_wtf.png', scaled_img_wtf)

    rotated_img = rotate_img(source_img, angle=45)
    cv2.imwrite(f'{render_dir}rotated_img.png', rotated_img)

    rotated_img_center = rotate_img_center(source_img, angle=45)
    cv2.imwrite(f'{render_dir}rotated_img_center.png', rotated_img_center)

    beveled_img = bevel_img(source_img, 0.3)
    cv2.imwrite(f'{render_dir}beveled_img.png', beveled_img)

    # эти две вместе не работают, только по отдельности
    piecewiselinear2 = piecewise_linear_mapping(source_img, 2)
    # piecewiselinear05 = piecewise_linear_mapping(source_img, 0.5)
    cv2.imwrite(f'{render_dir}piecewiselinear2.png', piecewiselinear2)
    # cv2.imwrite(f'{render_dir}piecewiselinear05.png', piecewiselinear05)

    projected_img = projection(source_img, [[1.1, 0.35, 0], [0.2, 1.1, 0], [0.00075, 0.0005, 1]])
    cv2.imwrite(f'{render_dir}projected_img.png', projected_img)

    T = [[0, 0], [1, 0], [0, 1], [0.00001, 0], [0.002, 0], [0.001, 0]]
    m = polynomial(source_img, T)
    cv2.imwrite(f'{render_dir}polynomial.png', m)

    sinusoid = sinusoidal(source_img)
    cv2.imwrite(f'{render_dir}sinusoid.png', sinusoid)

    barreled = barrel_distorsion(source_img)
    cv2.imwrite(f'{render_dir}barrel.png', barreled)

    src2 = cv2.imread('tech-vision/source/barrel.png')
    undistort_bar = undistort_barrel(src2)
    cv2.imwrite(f'{render_dir}undistort_bar.png', undistort_bar)

    p = pincushion_distorsion(source_img)
    cv2.imwrite(f'{render_dir}pil_distort.png', p)

    src3 = cv2.imread('tech-vision/source/pillow.png')
    undistort_pin = undistort_pincushion(src3)
    cv2.imwrite(f'{render_dir}undistort_pin.png', undistort_pin)
    
    f_part = cv2.imread('tech-vision/source/top.jpg', cv2.IMREAD_COLOR)
    s_part = cv2.imread('tech-vision/source/bot.jpg', cv2.IMREAD_COLOR)

    ans = join_img(f_part, s_part)
    cv2.imwrite(f'{render_dir}join.png', ans)

    I_3 = cv2.imread('tech-vision/source/I1.jpg', cv2.IMREAD_COLOR)
    I_2 = cv2.imread('tech-vision/source/I2.jpg', cv2.IMREAD_COLOR)
    I_1 = cv2.imread('tech-vision/source/I3.jpg', cv2.IMREAD_COLOR)
    status, I_stitch = auto_join_img([I_1, I_2, I_3])
    cv2.imwrite(f'{render_dir}auto_join.png', I_stitch)
