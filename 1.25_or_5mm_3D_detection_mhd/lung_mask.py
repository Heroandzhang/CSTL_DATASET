import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import SimpleITK
from skimage import measure, morphology

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2;
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num+1;
        slice_num = int(len(slices) / sec_num)
        slices.sort(key = lambda x:float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    # return np.array(image, dtype=np.int16), np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)
    return np.array(image, dtype=np.int16), np.array(
        [slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]], dtype=np.float32)

def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)
    
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
        
        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
        
    return bw

def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
        
    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
            
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
            
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)

def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
    
    return bw




def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw

def step1_python(case_path):
    case = load_scan(case_path)
    case_pixels, spacing = get_pixels_hu(case)
    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.2,8.2])
        # plt.imshow(bw[6], cmap=plt.cm.gray)
        # plt.show()
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    # plt.imshow(bw[6], cmap=plt.cm.gray)
    # plt.show()
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing
    
if __name__ == '__main__':
    # INPUT_FOLDER = 'E:/patient_id_dicom'
    # patients = os.listdir(INPUT_FOLDER)
    # patients.sort()
    # case_pixels, m1, m2, spacing = step1_python(os.path.join(INPUT_FOLDER,patients[6]))
    ###写入DICOM文件的路径
    origin_path = 'E:/patient_id_dicom'
    save_path='E:/seg_lung_DSB_int'
    img_save_path='E:/seg_image_DSB_int'
    # origin_path = 'D:\shortdateset\patient_id_mhd'
    # save_path='F:\long_1.25\seg_lung_DSB'
    # img_save_path='F:\long_1.25\seg_image_DSB'
    # origin_path = 'D:\shortdateset\patient_id_dicom'
    # save_path='D:\shortdateset\seg_lung_DSB'
    # img_save_path='D:\shortdateset\seg_image_DSB'

    # origin_path = 'E:/patient_id_dicom'
    dirlist = []
    for dirname in os.listdir(origin_path):  # 最原始的路径包含的文件夹
        origin_pathlist = os.path.join(origin_path, dirname)  # D:/桌面/1111\1 文件夹路径
        print(origin_pathlist)
        for dir2_dir in os.listdir(origin_pathlist):
            # print(dir2_dir)
            dir2_filelist = os.path.join(origin_pathlist, dir2_dir)
            # print(dir2_filelistt)
            if os.path.isdir(dir2_filelist) == True:
                dirlist.append(dir2_filelist)
            else:
                dirlist.append(origin_pathlist)
                break
    # print(dirlist)

    for i in range(len(dirlist)):
        dcm_dir = dirlist[i]
        # print(dcm_dir)
        lstFilesDCM = os.listdir(dcm_dir)

        RefDs = pydicom.read_file(os.path.join(dcm_dir, lstFilesDCM[0]))  # 读取第一张dicom图片

        # 第二步：得到dicom图片所组成3D图片的维度
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))  # ConstPixelDims是一个元组

        # 第三步：得到x方向和y方向的Spacing并得到z方向的层厚
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
        spacing = np.array(ConstPixelSpacing, dtype=float)[::-1]

        # 第四步：得到图像的原点
        Origin = RefDs.ImagePositionPatient
        ####DSB原始代码
        case_pixels, m1, m2, spacing = step1_python(dcm_dir)

        m1= m1.astype(float)

        m2=m2.astype(float)

        # print("原数组是:", m1)
        for x in np.nditer(m1, op_flags=['readwrite']):
            if x==1:
                x[...] =4
        # print('修改后的数组是：', m1)

        # print("原数组是:", m2)
        for x in np.nditer(m2, op_flags=['readwrite']):
            if x == 1:
                x[...] = 3
        # print('修改后的数组是：', m2)
        # plt.imshow(m1[6], cmap=plt.cm.gray)
        # plt.show()
        # plt.figure()
        # plt.imshow(m2[6], cmap=plt.cm.gray)
        # plt.show()

        lung_mask=m1+m2
        print(type(lung_mask))
        # plt.imshow(lung_mask[6], cmap=plt.cm.gray)
        # plt.show()

        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # if len(dcm_dir[20:]) > 2:
        #     a, b = os.path.split(dcm_dir[20:])
        #     c = a + '_' + b
        #     save_mhd_dir = os.path.join(save_path, "{}.mhd".format(c))
        #     # print(save_mhd_dir,'111')
        # else:
        #     save_mhd_dir = os.path.join(save_path, "{}.mhd".format(dcm_dir[20:]))
        # print(save_mhd_dir, 'save_mhd_dir')  ###保存文件
        print(dcm_dir[20:],'dcm_dir[14:]')
        if len(dcm_dir[20:]) > 2:
            a, b = os.path.split(dcm_dir[20:])
            c = a + '_' + b
            print(a,b,c,'aabbbcccccc')
            save_mhd_dir = os.path.join(save_path, "{}.mhd".format(str(c)))
            print(save_mhd_dir,'111')
        else:
            save_mhd_dir = os.path.join(save_path, "{}.mhd".format(dcm_dir[20:]))
            print(save_mhd_dir)
        print(save_mhd_dir, 'save_mhd_dir')  ###保存文件

        #将现在的numpy数组通过SimpleITK转化为mhd和raw文件
        lung_mask=lung_mask.astype(int)
        lung_mask=np.int16(lung_mask)

        sitk_img = SimpleITK.GetImageFromArray(lung_mask, isVector=False)
        sitk_img.SetSpacing(ConstPixelSpacing)
        sitk_img.SetOrigin(Origin)
        SimpleITK.WriteImage(sitk_img, save_mhd_dir)
        #
        #转化为png
        for i in range(lung_mask.shape[0]):
            print(dcm_dir[20:])
            arr = lung_mask[i, :, :]  # 获得第i张的单一数组type 'numpy.ndarray'
            img_save_dir = os.path.join(img_save_path, dcm_dir[20:])
            # print(img_save_dir, 'img_save_dir')
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)  ###建立文件夹，将图片保存到文件夹
            plt.imsave(os.path.join(img_save_dir, "{}_mask.png".format(i)), arr, cmap='gray')  # 定义命名规则，保存图片为彩色模式
            print('photo {} finished'.format(i))


###原始代码
#     first_patient = load_scan(INPUT_FOLDER + patients[25])
#     first_patient_pixels, spacing = get_pixels_hu(first_patient)
#     plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
#     plt.xlabel("Hounsfield Units (HU)")
#     plt.ylabel("Frequency")
#     plt.show()
    
#     # Show some slice in the middle
#     h = 80
#     plt.imshow(first_patient_pixels[h], cmap=plt.cm.gray)
#     plt.show()
    
#     bw = binarize_per_slice(first_patient_pixels, spacing)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()
    
#     flag = 0
#     cut_num = 0
#     while flag == 0:
#         bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num)
#         cut_num = cut_num + 1
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()
    
#     bw = fill_hole(bw)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()
    
#     bw1, bw2, bw = two_lung_only(bw, spacing)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()
