from PIL import Image
img=Image.open(r'/home/Chen_hongyu/lung_nodule_dataset/BMP/0001/12.bmp')
# 直接就输出图像的通道数了
print(len(img.split()))

