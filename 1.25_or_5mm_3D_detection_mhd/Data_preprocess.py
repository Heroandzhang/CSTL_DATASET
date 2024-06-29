path = '1.25_or_5mm_3D_detection_mhd'

#Read annotations.csv
annotations = pd.read_csv(path+'annotations.csv')
annotations.head()

#Import ".mhd"
mhd_list = glob.glob(path+'.mhd')

#Exclude paths, suffixes
presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

#Create the dictionary diameter_dict:
diameter_dict = {}
with open(path+'annotations.csv', "r") as f:
    for row in list(csv.reader(f))[1:]:
        series_uid = row[0]
        annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
        annotationDiameter_mm = float(row[4])

        diameter_dict.setdefault(series_uid, []).append(
            (annotationCenter_xyz, annotationDiameter_mm)
        )

#Print a CT scan
series_uid = '1'
mhd_path = glob.glob(
    path+'subset*/{}.mhd'.format(series_uid)
)[0]

ct_mhd = sitk.ReadImage(mhd_path)
ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

#Data type conversion from IRC to XYZ
import util  
util.irc2xyz

ct_a.clip(-1000, 1000, ct_a)  

series_uid = series_uid
hu_a = ct_a  

IrcTuple = collections.namedtuple('IrcTuple',['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple',['x', 'y', 'z'])
origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

#Taking out a nodule from a CT scan
def getRawCandidate(center_xyz, width_irc):
    center_irc = util.xyz2irc( 
        center_xyz,
        origin_xyz,
        vxSize_xyz,
        direction_a,
    )

    slice_list = []
    for axis, center_val in enumerate(center_irc):
        start_ndx = int(round(center_val - width_irc[axis]/2))
        end_ndx = int(start_ndx + width_irc[axis])

        assert center_val >= 0 and center_val < hu_a.shape[axis], repr([series_uid, center_xyz, origin_xyz, vxSize_xyz, center_irc, axis])

        if start_ndx < 0:
            # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
            #     series_uid, center_xyz, center_irc, hu_a.shape, width_irc))
            start_ndx = 0
            end_ndx = int(width_irc[axis])

        if end_ndx > hu_a.shape[axis]:
            # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
            #     series_uid, center_xyz, center_irc, hu_a.shape, width_irc))
            end_ndx = hu_a.shape[axis]
            start_ndx = int(hu_a.shape[axis] - width_irc[axis])

        slice_list.append(slice(start_ndx, end_ndx))

    ct_chunk = hu_a[tuple(slice_list)]

    return ct_chunk, center_irc

# The complete program encapsulated into a class:
# The previous steps were to break the following code into parts to be parsed separately, now write it as a complete class to import the data:
import torch
from torch.utils.data import Dataset
from logconf import logging
from util import XyzTuple, xyz2irc

log = logging.getLogger(__name__)

class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            path+'subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        ct_a.clip(-1000, 1000, ct_a)  # Hu的正常值为-1000~+1000

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_liast)]

        return ct_chunk, center_irc


# @functools.lru_cache(1, typed=True) #Stored in cache to avoid double loading of data
def getCt(series_uid):
    return Ct(series_uid)

# @raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


