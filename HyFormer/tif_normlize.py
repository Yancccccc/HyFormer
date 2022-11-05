from osgeo import gdal
import os,glob
import numpy as np



def read_img(img_path):
    dataset = gdal.Open(img_path)
    im_bands = dataset.RasterCount
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    del dataset
    return im_proj, im_geotrans, im_data

def write_img(filename,im_proj,im_geotrans,im_data):
  if 'int8' in im_data.dtype.name:
    datatype = gdal.GDT_Byte
  elif 'int16' in im_data.dtype.name:
    datatype = gdal.GDT_UInt16
  else:
    datatype = gdal.GDT_Float32

    #判读数组维数
  if len(im_data.shape) == 3:
    im_bands, im_height, im_width = im_data.shape
  else:
    im_bands, (im_height, im_width) = 4,im_data.shape

    #创建文件
  driver = gdal.GetDriverByName("GTiff")      #数据类型必须有，因为要计算需要多大内存空间
  dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

  dataset.SetGeoTransform(im_geotrans)       #写入仿射变换参数
  dataset.SetProjection(im_proj)          #写入投影

  if im_bands == 1:
    dataset.GetRasterBand(1).WriteArray(im_data) #写入数组数据
  else:
    for i in range(im_bands):
      dataset.GetRasterBand(i+1).WriteArray(im_data[i])
  del dataset


def readTif(imgPath, bandsOrder=[1, 2, 3, 4]):
  dataset = gdal.Open(imgPath, gdal.GA_ReadOnly)
  bands = dataset.RasterCount
  cols = dataset.RasterXSize
  rows = dataset.RasterYSize
  channel = dataset.RasterCount
  data = np.empty([channel,rows, cols], dtype=float)
  for i in range(len(bandsOrder)):
    band = dataset.GetRasterBand(bandsOrder[i])
    oneband_data = band.ReadAsArray()
    data[i, :, :] = oneband_data
  return data

def stretchImg(imgPath, resultPath, lower_percent=0.5, higher_percent=99.5):
  RGB_Array = readTif(imgPath)
  band_Num = RGB_Array.shape[0]
  JPG_Array = np.zeros_like(RGB_Array, dtype=np.uint8)
  for i in range(band_Num):
    minValue = 0
    maxValue = 255
      # 获取数组RGB_Array某个百分比分位上的值
    low_value = np.percentile(RGB_Array[i,:, :], lower_percent)
    high_value = np.percentile(RGB_Array[i,:, :], higher_percent)
    temp_value = minValue + (RGB_Array[i,:, :] - low_value) * (maxValue - minValue) / (high_value - low_value)
    temp_value[temp_value < minValue] = minValue
    temp_value[temp_value > maxValue] = maxValue
    JPG_Array[i,:, :] = temp_value
  return JPG_Array

if __name__ == "__main__":
  #os.chdir('data')
  proj, geotrans, data = read_img("data/changxing/2022/20220727.tif")
  JPG_Array =stretchImg("data/changxing/2022/20220727.tif", "data", lower_percent=0.5, higher_percent=99.5)
  write_img('data/changxing/2022/20220727_pixel_255.tif', proj, geotrans, JPG_Array)

  #os.chdir(r'D:\Python_Practice')            #切换路径到待处理图像所在文件夹

