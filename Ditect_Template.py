import cv2
import time
import sys
import os
import functools
import numpy as np
import glob
from PIL import ImageFont, ImageDraw, Image, ImageEnhance
print = functools.partial(print, flush=True)
import random
import math
import threading
import os.path
import pickle
from concurrent.futures import ThreadPoolExecutor

cv2.setUseOptimized(True)


def calc_black_whiteArea_2nd(bw_image):
    threshold = 127  # 二値化閾値
    img_blur = cv2.blur(bw_image, (9, 9))

    _, img_binary = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_BINARY)

    pixel_number = img_binary.size
    white_pixel_number = cv2.countNonZero(img_binary)
    black_pixel_number = pixel_number - white_pixel_number

    return white_pixel_number, black_pixel_number


def calc_whiteArea(bw_image):
    image_size = bw_image.size
    whitePixels = cv2.countNonZero(bw_image)
    blackPixels = bw_image.size - whitePixels

    whiteAreaRatio = (whitePixels / image_size) * 100  # [%]
    blackAreaRatio = (blackPixels / image_size) * 100  # [%]

    return whiteAreaRatio


def enlarge_region_around_center(points, scale_factor):
    # pointsは4つの点の座標を含むリスト [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    # scale_factorは拡大率

    # 中心座標を計算
    center_x = sum(x for x, y in points) / 4
    center_y = sum(y for x, y in points) / 4

    # 各点を中心を基準に拡大
    enlarged_points = [
        (
            int(center_x + (x - center_x) * scale_factor),
            int(center_y + (y - center_y) * scale_factor)
        )
        for x, y in points
    ]

    return enlarged_points


# 四角形の中に点列が幾つ含まれているかの判定
def count_inside(points, test_points, Diff_ImageSize):
    expSize = 1.1
    if Diff_ImageSize < 10:
        expSize = 1.2
    elif Diff_ImageSize < 50:
        expSize = 1.3
    elif Diff_ImageSize < 500:
        expSize = 1.4
    elif Diff_ImageSize < 2000:
        expSize = 1.5
    else:
        expSize = 2

    # 領域を拡大
    point_list_exp = enlarge_region_around_center(points, expSize)

    min_x = min(p[0] for p in point_list_exp)
    max_x = max(p[0] for p in point_list_exp)
    min_y = min(p[1] for p in point_list_exp)
    max_y = max(p[1] for p in point_list_exp)

    count = sum(min_x <= x <= max_x and min_y <= y <= max_y for x, y in test_points)
    return count


# 原点からの距離を計算
def distance_from_origin(point):
    """原点からの距離を計算する関数"""
    return math.sqrt(point[0] ** 2 + point[1] ** 2)


# 特徴点検出器
def detector(name):
    if name == "AgastFeature":
        detector = cv2.AgastFeatureDetector_create()
    elif name == "SIFT":
        detector = cv2.SIFT_create()
    elif name == "MSER":
        detector = cv2.MSER_create()
    elif name == "AKAZE":
        detector = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=0.001,
            nOctaves=4,
            nOctaveLayers=4,
            diffusivity=cv2.KAZE_DIFF_PM_G2,
        )
    elif name == "BRISK":
        detector = cv2.BRISK_create(thresh=10, octaves=2)
    elif name == "KAZE":
        detector = cv2.KAZE_create()
    elif name == "ORB":
        detector = cv2.ORB_create()
    else:
        detector = cv2.SimpleBlobDetector_create()
    return detector


# ラベルテーブルの情報を元に入力画像に色をつける
def put_color_to_objects(src_img, label_table):
    label_img = np.zeros_like(src_img)
    print(str(label_table.max()))
    for label in range(label_table.max() + 1):
        label_group_index = np.where(label_table == label)
        label_img[label_group_index] = random.sample(range(255), k=3)
    return label_img


# 点が中に入っているかの確認
def inrect(startPoint_X, startPoint_Y, height, width, startPoint_X_diff, startPoint_Y_diff):
    a = (startPoint_X, startPoint_Y)
    b = (startPoint_X + width, startPoint_Y)
    c = (startPoint_X + width, startPoint_Y + height)
    d = (startPoint_X, startPoint_Y + height)
    e = (startPoint_X_diff, startPoint_Y_diff)

    # 原点から点へのベクトルを求める
    vector_a = np.array(a)
    vector_b = np.array(b)
    vector_c = np.array(c)
    vector_d = np.array(d)
    vector_e = np.array(e)

    # 点から点へのベクトルを求める
    vector_ab = vector_b - vector_a
    vector_ae = vector_e - vector_a
    vector_bc = vector_c - vector_b
    vector_be = vector_e - vector_b
    vector_cd = vector_d - vector_c
    vector_ce = vector_e - vector_c
    vector_da = vector_a - vector_d
    vector_de = vector_e - vector_d

    # 外積を求める
    vector_cross_ab_ae = np.cross(vector_ab, vector_ae)
    vector_cross_bc_be = np.cross(vector_bc, vector_be)
    vector_cross_cd_ce = np.cross(vector_cd, vector_ce)
    vector_cross_da_de = np.cross(vector_da, vector_de)

    return vector_cross_ab_ae < 0 and vector_cross_bc_be < 0 and vector_cross_cd_ce < 0 and vector_cross_da_de < 0


# 画像表示
def imshow(img):
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_files_with_pattern(directory, pattern, infiles):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern in file:
                full_path = os.path.join(root, file)
                infiles.append(full_path)


def get_parent_folder_name(full_path):
    # パスから親ディレクトリのパスを取得
    parent_directory = os.path.dirname(full_path)

    # 親ディレクトリの名前だけを取得
    parent_name = os.path.basename(parent_directory)

    return parent_name


def getExpSize(img):
    height, width = img.shape[:2]
    ex_temp_img1 = 1

    if (height * width) < 10000:
        ex_temp_img1 = 2
    elif (height * width) < 30000:
        ex_temp_img1 = 2
    elif (height * width) < 100000:
        ex_temp_img1 = 1
    elif (height * width) < 200000:
        ex_temp_img1 = 1 / 2
    else:
        ex_temp_img1 = 1 / 4

    return ex_temp_img1


class TemplateInfo:
    def __init__(self, templatePath, skipKaze=True):
        self.templatePath = templatePath
        self.basename = os.path.splitext(os.path.basename(templatePath))[0]
        self.Image = None
        self.gray = None
        self.kp1 = None
        self.des1 = None
        self.kp1_SHIFT = None
        self.des1_SHIFT = None
        self.detector = None
        self.detector_SHIFT = None
        self.detectorNo = 0
        self.detectRctp1 = None
        self.detectRctp2 = None
        self.startPoint_X = None
        self.startPoint_Y = None
        self.detectcheck = False
        self.detectScoar = False
        self.whiteAreaRatio = None
        self.blackAreaRatio = False
        self.labelsNum = 0
        self.ScoarRect = 0
        self.height = None
        self.width = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ex_temp_img1 = 0

        self.Image = imread(templatePath)

        akaze = detector("AKAZE")
        brisk = detector("BRISK")
        sift = detector("SIFT")
        kase = detector("SIFT")
        orb = detector("ORB")

        self.detector_SHIFT = sift

        whitespace = 0
        self.height, self.width = self.Image.shape[:2]
        template_img = np.ones((self.height + whitespace * 2, self.width + whitespace * 2, 3), np.uint8) * 255
        template_img[whitespace:whitespace + self.height, whitespace:whitespace + self.width] = self.Image

        # 拡大
        ex_temp_img1 = getExpSize(self.Image)
        self.ex_temp_img1 = ex_temp_img1

        img1_exp = cv2.resize(template_img, None, interpolation=cv2.INTER_LINEAR, fx=ex_temp_img1, fy=ex_temp_img1)

        self.gray = cv2.cvtColor(img1_exp, cv2.COLOR_BGR2GRAY)

        kp1_a, des1_a = akaze.detectAndCompute(self.gray, None)
        kp1_b, des1_b = brisk.detectAndCompute(self.gray, None)
        kp1_o, des1_o = orb.detectAndCompute(self.gray, None)
        kp1_s, des1_s = sift.detectAndCompute(self.gray, None)
        kp1_k, des1_k = kase.detectAndCompute(self.gray, None)

        self.kp1_SHIFT = kp1_s
        self.des1_SHIFT = des1_s

        self.detector = brisk

        detectCount = len(kp1_b)
        print("AKAZE特徴点の数:" + str(len(kp1_a)))
        print("BRISK特徴点の数:" + str(len(kp1_b)))
        print("ORB特徴点の数:" + str(len(kp1_o)))
        print("SIFT特徴点の数:" + str(len(kp1_s)))
        print("KAZE特徴点の数:" + str(len(kp1_k)))

        # 基本はbrisk採用だが、特徴点が20より小さい場合は別ロジックを採用
        if len(kp1_b) < 20:
            if len(kp1_s) > len(kp1_b):
                self.detector = sift
                self.detectorNo = 3
                detectCount = len(kp1_s)
                self.bf = cv2.BFMatcher()

        if detectCount < 5 and skipKaze is False:
            if len(kp1_k) > 5:
                self.detector = kase
                self.detectorNo = 4

        self.kp1, self.des1 = self.detector.detectAndCompute(self.gray, None)

        print(self.detectorNo)

        # 差分画像をグレースケールに変換
        gray_diff = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        # 2値化処理を行う
        self.ScoarRect = self.calc_ScoarRect(gray_diff)

    def calc_ScoarRect(self, gray_diff):
        ScoarRect = 0
        height_check, width_chec = gray_diff.shape[:2]

        if height_check > 10 and width_chec > 10:
            # 2値化処理を行う
            _, thresh = cv2.threshold(gray_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

            edge_image = cv2.Canny(thresh, 50, 70)
            kernel = np.ones((3, 3), np.uint8)
            edge_image = cv2.dilate(edge_image, kernel, iterations=3)

            # 輪郭を検出する
            contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

            contours = list(filter(lambda x: cv2.contourArea(x) > 10, contours))

            # 輪郭を描画する
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                ScoarRect = ScoarRect + (w * h)

        return ScoarRect

    def checkRectPoint(self, matches, kp1, kp2, ex_temp_img1, ex_temp_img2, startIndex=0):
        StartP = kp1[matches[startIndex].queryIdx].pt
        StartP2 = kp2[matches[startIndex].trainIdx].pt

        RectList = 0
        startPoint_X = StartP2[0] / ex_temp_img2 - StartP[0] / ex_temp_img1
        startPoint_Y = StartP2[1] / ex_temp_img2 - StartP[1] / ex_temp_img1
        for matche in matches:
            StartP = kp1[matche.queryIdx].pt
            StartP2 = kp2[matche.trainIdx].pt

            startPoint_X_diff = StartP2[0] / ex_temp_img2 - StartP[0] / ex_temp_img1
            startPoint_Y_diff = StartP2[1] / ex_temp_img2 - StartP[1] / ex_temp_img1

            if inrect(startPoint_X, startPoint_Y, self.height, self.width, startPoint_X_diff, startPoint_Y_diff):
                RectList = RectList + 1

        if RectList > 0:
            print(str(self.basename) + "  :" + str(RectList))


class Imagecache:
    def __init__(self):
        # 初期化
        self.Global_kp2 = {}
        self.Global_des2 = {}
        self.Global_Size = {}

    def clear(self):
        self.Global_kp2.clear()
        self.Global_des2.clear()
        self.Global_Size.clear()


class Imagecache_List:
    def __init__(self):
        # 初期化
        self.ImagecacheL = {}

        for ii in (2, 1, 1 / 2, 1 / 4):
            self.ImagecacheL[ii] = Imagecache()

    def clear(self):
        for key, value in self.ImagecacheL.items():
            value.clear()


class TemplateResult:
    def __init__(self):
        self.basename = ""
        self.detectRctp1 = None
        self.detectRctp2 = None
        self.startPoint_X = None
        self.startPoint_Y = None
        self.ScoarRect = 0
        self.height = None
        self.width = None
        self.detectScoar = 0


# 特徴点判定処理
def Do_detector(img2, ex_temp_img1, ex_temp_img2, TemplateInfo, counter, ImagecacheImp):
    img2_exp = cv2.resize(img2, None, interpolation=cv2.INTER_LINEAR, fx=ex_temp_img2, fy=ex_temp_img2)

    # img2をグレースケールで読み出し
    gray2 = cv2.cvtColor(img2_exp, cv2.COLOR_BGR2GRAY)

    detector = TemplateInfo.detector

    if counter == 0:
        ImagecacheImp.clear()

    if TemplateInfo.detectorNo not in ImagecacheImp.ImagecacheL[ex_temp_img2].Global_kp2 or ImagecacheImp.ImagecacheL[ex_temp_img2].Global_Size[TemplateInfo.detectorNo] != ex_temp_img2:
        ImagecacheImp.ImagecacheL[ex_temp_img2].Global_kp2[TemplateInfo.detectorNo], ImagecacheImp.ImagecacheL[ex_temp_img2].Global_des2[TemplateInfo.detectorNo] = detector.detectAndCompute(gray2, None)
        ImagecacheImp.ImagecacheL[ex_temp_img2].Global_Size[TemplateInfo.detectorNo] = ex_temp_img2

    cv2.destroyAllWindows()

    return (
        TemplateInfo.kp1,
        TemplateInfo.des1,
        ImagecacheImp.ImagecacheL[ex_temp_img2].Global_kp2[TemplateInfo.detectorNo],
        ImagecacheImp.ImagecacheL[ex_temp_img2].Global_des2[TemplateInfo.detectorNo],
        img2_exp,
        gray2,
    )


def filter_matches_by_ransac(keypoints1, keypoints2, matches, ransac_thresh=5.0):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    filtered_matches = []
    mask_M = None
    mask = None
    if len(src_pts) > 4 and len(dst_pts) > 4:
        try:
            # RANSACを使用して一致点をフィルタリング
            mask_M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

            # フィルタリングされたマッチングのインデックス
            filtered_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]
        except ValueError:
            return filtered_matches, mask_M, mask

    return filtered_matches, mask_M, mask


def find_min_max_coordinates(point_list):
    x_coords, y_coords = zip(*point_list)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return (max_x - min_x) * (max_y - min_y)


# 領域を倍率変更する
def halve_region(x, y, width, height, expSize):
    # 領域の中心座標を計算
    center_x = x + (width / 2)
    center_y = y + (height / 2)

    # 領域の幅と高さを半分にする
    new_width = width * expSize
    new_height = height * expSize

    # 新しい領域の左上隅の座標を計算
    new_x = center_x - (new_width / 2)
    new_y = center_y - (new_height / 2)

    if new_x < 0:
        new_x = 0

    if new_y < 0:
        new_y = 0

    return new_x, new_y, new_width, new_height


def calculate_ImagePoint(imageSrc, imageDst, startPoint_X, startPoint_Y):
    # 画像の高さと幅を取得
    height, width = imageSrc.shape

    # 画像の端の4点の座標を計算
    top_left = (startPoint_X, startPoint_Y)
    top_right = (startPoint_X + width - 1, startPoint_Y)
    bottom_left = (startPoint_X, startPoint_Y + height - 1)
    bottom_right = (startPoint_X + width - 1, startPoint_Y + height - 1)

    src_pts = np.array([[top_left], [bottom_left], [bottom_right], [top_right]], dtype=np.float32)

    # 画像の高さと幅を取得
    height, width = imageDst.shape

    # 画像の端の4点の座標を計算
    top_left = (0, 0)
    top_right = (width - 1, 0)
    bottom_left = (0, height - 1)
    bottom_right = (width - 1, height - 1)

    dst_pts = np.array([[top_left], [bottom_left], [bottom_right], [top_right]], dtype=np.float32)

    return src_pts, dst_pts


def calculate_PerspectiveTransform(imageSrc, imageDst, startPoint_X, startPoint_Y):
    src_pts, dst_pts = calculate_ImagePoint(imageSrc, imageDst, startPoint_X, startPoint_Y)

    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return mat


def generate_heatmap(points, image_width, image_height, divisions, show=False):
    # 画像を指定された数のセルに分割
    grid = np.zeros((divisions, divisions))
    badindex = True

    for x, y in points:
        y_index = int(y / (image_height / divisions))
        x_index = int(x / (image_width / divisions))
        if y_index > divisions - 1 or y_index < 0 or x_index > divisions - 1 or x_index < 0:
            badindex = False

    if badindex is True:
        for x, y in points:
            grid[int(y / (image_height / divisions)), int(x / (image_width / divisions))] += 1

        if np.sum(grid) > 0:
            # 各セルの合計値を正規化
            grid /= np.sum(grid)
        else:
            badindex = False

    return badindex, grid


def calculate_overlap(heatmap1, heatmap2):
    # NumPy配列に変換する
    heatmap1_np = np.array(heatmap1)
    heatmap2_np = np.array(heatmap2)

    # 二つのヒートマップの要素ごとの差の絶対値を計算
    absolute_difference = np.abs(heatmap1_np - heatmap2_np)

    # 一致率を計算
    overlap_percentage = 1 - np.sum(absolute_difference) / 2

    return overlap_percentage


def calculate_HeatMap(pts1, pts2, img_trim, img1, startPoint_X_exp, startPoint_Y_exp, multipleValue):
    mat = calculate_PerspectiveTransform(img_trim, img1, startPoint_X_exp, startPoint_Y_exp)

    # 射影変換を行う
    transformed_points = cv2.perspectiveTransform(pts2.reshape(1, -1, 2), mat)
    transformed_points = transformed_points.reshape(-1, 2)

    height, width = img1.shape

    # 16分割のヒートマップを生成
    rt_pst, heatmap_pst = generate_heatmap(pts1, width, height, multipleValue)
    rt_dst, heatmap_dst = generate_heatmap(transformed_points, width, height, multipleValue)

    # 一致率を計算
    overlap_percentage = calculate_overlap(heatmap_pst, heatmap_dst)

    if rt_pst is False or rt_dst is False:
        overlap_percentage = -1

    return overlap_percentage


def distance(point1, point2):
    """2点間の距離を計算する関数"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def has_close_point(point_sequence, target_point, threshold_distance):
    for point in point_sequence:
        if distance(point, target_point) < threshold_distance:
            return True

    return False


# 画像一致判定
def Do_detector_main(img2, TemplateInfo, counter, ImagecacheImp, TemplateResultList):
    TemplateInfo.detectcheck = False
    multipleValue = 6

    # 拡大/縮小
    ex_temp_imgSize = TemplateInfo.ex_temp_img1

    kp1, des1, kp2, des2, img2_exp, gray2 = Do_detector(img2, ex_temp_imgSize, ex_temp_imgSize, TemplateInfo, counter, ImagecacheImp)

    if len(kp1) > 0 and len(des1) > 0 and len(kp2) > 0 and len(des2) > 0:
        knn = TemplateInfo.bf.knnMatch(des1, des2, k=2)
        matches_akaze = TemplateInfo.bf.match(des1, des2)

        goodcount = 0
        # Apply ratio test
        good = []
        img2_pt_List = []
        try:
            for m, n in knn:
                if m.distance < 0.8 * n.distance:
                    goodcount = goodcount + 1
                    good.append([m])
                    img2_pt = kp2[m.trainIdx].pt
                    img2_pt_List.append(img2_pt)
        except ValueError:
            return

        matches_org = sorted(matches_akaze, key=lambda x: x.distance)

        # RANSACによるフィルタリング
        filtered_matches, mask_M, mask = filter_matches_by_ransac(kp1, kp2, matches_akaze)

        height, width = TemplateInfo.gray.shape[:2]
        img2_height, img2_width = img2_exp.shape[:2]

        Diff_ImageSize = (img2_height * img2_width) / (height * width)
        filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)

        # 一致した点の座標を取得
        pts1_org = np.array([kp1[match.queryIdx].pt for match in matches_org], dtype=np.float32)
        pts1 = np.array([kp1[match.queryIdx].pt for match in filtered_matches], dtype=np.float32)
        pts2 = np.array([kp2[match.trainIdx].pt for match in filtered_matches], dtype=np.float32)

        count_ii = 0
        count_insideNum_range = 0
        startPoint_X_t = 10
        startPoint_Y_t = 10
        width_ext = 0
        height_ext = 0

        img2_pt_List_org = []

        img_trim_org = None
        img_trim_pre = None

        for ii in filtered_matches:
            img2_pt_List_org.append(kp2[ii.trainIdx].pt)

        count_ii = 0
        count_insideNum_range = 0
        startPoint_X_t = 10
        startPoint_Y_t = 10
        width_ext = 0
        height_ext = 0
        overlap_percentage_pre = 0
        Filterd_pt_List = []

        for ii in filtered_matches:
            StartP = kp1[filtered_matches[count_ii].queryIdx].pt
            StartP2 = kp2[filtered_matches[count_ii].trainIdx].pt

            startPoint_X = StartP2[0] - StartP[0]
            startPoint_Y = StartP2[1] - StartP[1]

            closecheck = has_close_point(Filterd_pt_List, (startPoint_X, startPoint_Y), 3)
            if closecheck is True:
                continue

            Filterd_pt_List.append((startPoint_X, startPoint_Y))

            if startPoint_X < 0:
                startPoint_X = 0
            if startPoint_Y < 0:
                startPoint_Y = 0

            # 輪郭を描画する。
            rect_width = int(startPoint_X + int(width))
            rect_height = int(startPoint_Y + int(height))
            if img2_width < rect_width:
                rect_width = img2_width
            if img2_height < rect_height:
                rect_height = img2_height

            img_trim = img2_exp[int(startPoint_Y):rect_height, int(startPoint_X):rect_width]
            expSize = 1.3

            if count_ii == 0:
                startPoint_X_t = startPoint_X
                startPoint_Y_t = startPoint_Y
                width_ext = rect_width
                height_ext = rect_height
                img_trim_org = img_trim
                img_trim_pre = img_trim

            count_ii = count_ii + 1

            for jj in range(10):
                startPoint_X_exp, startPoint_Y_exp, width_ext_tmp, height_ext_tmp = halve_region(startPoint_X, startPoint_Y, width, height, expSize)

                rect_width_ext = int(startPoint_X_exp + int(width_ext_tmp))
                rect_height_ext = int(startPoint_Y_exp + int(height_ext_tmp))

                if img2_width < rect_width_ext or img2_height < rect_height_ext:
                    rect_width_ext = img2_width
                    rect_height_ext = img2_height

                img_trim_2 = img2_exp[int(startPoint_Y_exp):int(rect_height_ext), int(startPoint_X_exp):int(rect_width_ext)]

                height_check, width_chec = img_trim_2.shape[:2]
                if height_check <= 0 or width_chec <= 0:
                    continue

                pointsRect = [
                    (int(startPoint_X_exp), int(startPoint_Y_exp)),
                    (int(startPoint_X_exp) + int(width_ext_tmp), int(startPoint_Y_exp)),
                    (int(startPoint_X_exp), int(startPoint_Y_exp) + int(height_ext_tmp)),
                    (int(startPoint_X_exp) + int(width_ext_tmp), int(startPoint_Y_exp) + int(height_ext_tmp)),
                ]
                count_insideNum_tmp = count_inside(pointsRect, img2_pt_List_org, Diff_ImageSize)

                overlap_percentage = calculate_HeatMap(pts1, pts2, img_trim_2, TemplateInfo.gray, startPoint_X_exp, startPoint_Y_exp, multipleValue)

                if overlap_percentage == -1:
                    break

                if overlap_percentage_pre <= overlap_percentage:
                    overlap_percentage_pre = overlap_percentage
                    startPoint_X_t = startPoint_X_exp
                    startPoint_Y_t = startPoint_Y_exp
                    width_ext = width_ext_tmp
                    height_ext = height_ext_tmp
                    img_trim_pre = img_trim_org
                    img_trim_org = img_trim_2
                    count_insideNum_range = count_insideNum_tmp

                expSize = expSize - 0.1

        height, width = TemplateInfo.gray.shape
        # 16分割のヒートマップを生成
        rt_org, heatmap_pst_org = generate_heatmap(pts1_org, width, height, multipleValue, True)
        rt_, heatmap_pst = generate_heatmap(pts1, width, height, multipleValue, True)

        # 一致率を計算
        overlap_percentage_org = calculate_overlap(heatmap_pst, heatmap_pst_org) * 100
        print(f"ヒートマップの一致率(オリジナル同士): {overlap_percentage_org :.2f}%")

        overlap_percentage = overlap_percentage_pre * 100
        print(f"ヒートマップの一致率: {overlap_percentage :.2f}%")

        TemplateInfo.detectRctp1 = (int(startPoint_X_t / ex_temp_imgSize), int(startPoint_Y_t / ex_temp_imgSize))
        TemplateInfo.detectRctp2 = (
            int(startPoint_X_t / ex_temp_imgSize + int(width_ext / ex_temp_imgSize)),
            int(startPoint_Y_t / ex_temp_imgSize + int(height_ext / ex_temp_imgSize)),
        )
        TemplateInfo.startPoint_X = int(startPoint_X_t / ex_temp_imgSize)
        TemplateInfo.startPoint_Y = int(startPoint_Y_t / ex_temp_imgSize)

        print("goodcount:  " + str(goodcount))
        print("matches:  " + str(len(matches_org)))
        print("count_insideNum_range:  " + str(count_insideNum_range))
        print("filtered_matches:  " + str(len(filtered_matches)))

        # 16分割なので特徴点が少ない場合は閾値を70⇒80に上げる
        threshold_judge = 75
        if count_insideNum_range < 10:
            threshold_judge = 85
        TemplateInfo.detectScoar = (overlap_percentage_org + overlap_percentage) / 2
        # 一致距離が0.8以内の箇所が1個以上ある事、またはオリジナル同士の劣化が70%以上かつ配置が80%以上であれば一致とみなす
        if goodcount > 1 and TemplateInfo.detectScoar > threshold_judge and count_insideNum_range > 4:
            TemplateInfo.detectcheck = True
            print("Find Template:  " + TemplateInfo.basename)

            ResultData = TemplateResult()
            ResultData.detectcheck = True
            ResultData.frameNo = TemplateInfo.frameNo
            ResultData.detectRctp1 = TemplateInfo.detectRctp1
            ResultData.detectRctp2 = TemplateInfo.detectRctp2
            ResultData.basename = TemplateInfo.basename
            ResultData.startPoint_X = TemplateInfo.startPoint_X
            ResultData.startPoint_Y = TemplateInfo.startPoint_Y
            ResultData.detectScoar = TemplateInfo.detectScoar
            TemplateResultList.append(ResultData)


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def Template_check(path, img2, TemplateList, ImagecacheImp):
    cur_dir = os.path.join(os.getcwd(), "template")

    counter = 0
    for p in glob.glob(os.path.join(cur_dir, r"*.png")):
        Do_detector_main(img2, TemplateList[counter], counter, ImagecacheImp)
        counter = counter + 1


def Template_init(path, TemplateList):
    cur_dir = os.path.join(os.getcwd(), "template")

    for p in glob.glob(os.path.join(cur_dir, r"*.png")):
        template = imread(p)
        TemplateList.append(TemplateInfo(p))


def TemplateResultOut(frame, video_red, TemplateList):
    for TemplateInfo in TemplateList:
        if TemplateInfo.detectcheck is True:
            cv2.rectangle(frame, TemplateInfo.detectRctp1, TemplateInfo.detectRctp2, color=(0, 0, 255), thickness=2)
            result = "{0} ({1:.3f})".format(TemplateInfo.basename, TemplateInfo.detectScoar)
            fontpath = "C:\\Windows\\Fonts\\HGRPP1.TTC"  # Windows10 だと C:\Windows\Fonts\ 以下にフォントがあります。
            font = ImageFont.truetype(fontpath, 18)

            img_pil = Image.fromarray(frame)

            draw = ImageDraw.Draw(img_pil)

            position = (TemplateInfo.startPoint_X, TemplateInfo.startPoint_Y - 5)
            draw.text(position, result, font=font)

            frame = np.array(img_pil)

    video_red.write(frame)


def frame_Thread_main(cur_dir, n, frame, TemplateList, skip=False):
    counter = 0
    ImagecacheImp = Imagecache_List()
    ImagecacheImp.clear()
    TemplateResultList = []
    if skip is False:
        for p in glob.glob(os.path.join(cur_dir, r"*.png")):
            TemplateList[counter].frameNo = n
            Do_detector_main(frame, TemplateList[counter], counter, ImagecacheImp, TemplateResultList)
            counter = counter + 1

    return frame, TemplateResultList


def save_all_frames(video_path, dir_path, video_red, basename, TemplateList, ext='jpg', skip=1):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0
    beforeFrame = None
    firstFrameFlag = 0
    TemplateList_Before = []
    whiteArea = 100
    whiteArealimit = 3
    threadn = cv2.getNumberOfCPUs()
    executor = ThreadPoolExecutor(max_workers=max(threadn - 2, 1))
    pending = []

    template_dir = os.path.join(os.getcwd(), "template")

    while True:
        ret, frame = cap.read()

        # フレームが取得できない場合はループを抜ける
        if not ret:
            # タスクの完了を待機
            executor.shutdown()

            while len(pending) > 0:
                if pending[0].done():
                    frame_R, TemplateList_R = pending.pop(0).result()
                    TemplateResultOut(frame_R, video_red, TemplateList_R)
            return

        # 取り込んだフレームに対して差分をとって動いているところが明るい画像を作る
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if firstFrameFlag == 0:
            beforeFrame = gray.copy().astype('float')
            firstFrameFlag = 1
        else:
            # 現フレームと前フレームの加重平均を使うと良いらしい
            cv2.accumulateWeighted(gray, beforeFrame, 0.2)
            mdframe = cv2.absdiff(gray, cv2.convertScaleAbs(beforeFrame))
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), 'jpg'), mdframe)

            # 動いているエリアの面積を計算してちょうどいい検出結果を抽出する
            thresh = cv2.threshold(mdframe, 0, 255, cv2.THRESH_OTSU)[1]

            beforeFrame = gray.copy().astype('float')
            whiteArea, blackArea = calc_black_whiteArea_2nd(thresh)

        if whiteArea >= whiteArealimit:
            future = executor.submit(frame_Thread_main, template_dir, n, frame.copy(), TemplateList)
            pending.append(future)
        else:
            future = executor.submit(frame_Thread_main, template_dir, n, frame.copy(), TemplateList, True)
            pending.append(future)

        n = n + 1
        print("loop:" + str(n))


# 処理の開始時間を記録
start_time = time.time()

movie = cv2.VideoCapture('sample_video.mp4')

fps = int(movie.get(cv2.CAP_PROP_FPS))  # 動画のFPSを取得
w = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))  # 動画の横幅を取得
h = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 動画の縦幅を取得
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 動画保存時のfourcc設定（mp4用）

codec = cv2.VideoWriter_fourcc(*'mp4v')
movie.release()

video_red = cv2.VideoWriter('check_video.mp4', codec, fps, (w, h))

TemplateList = []
Template_init("./template", TemplateList)

save_all_frames('sample_video.mp4', 'frame/result_mattching', video_red, 'sample_video_img', TemplateList, 'png', 1)

video_red.release()

# 処理の終了時間を記録
end_time = time.time()

# 経過時間を計算
elapsed_time = end_time - start_time

print("処理時間:", elapsed_time, "秒")
