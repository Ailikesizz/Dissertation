import pandas as pd
import gdal

df=pd.read_csv('/data/shangrui/geom_c.csv')


for i in df.index:
      if df['income_adjusted'][i]=='wealthy':
        imagery_filename = "/data/shangrui/c/wealthy/%s.tif"%(df['tower'][i])
        gdal.Warp(imagery_filename,
        'google_tms.xml',
        outputBounds=[df['st_xmin'][i],df['st_ymin'][i],df['st_xmax'][i],df['st_ymax'][i]],
        dstSRS='EPSG:3857')
      elif df['income_adjusted'][i]=='average':
         imagery_filename = '/data/shangrui/c/average/%s.tif'%(df['tower'][i])
         gdal.Warp(
          imagery_filename,
          'google_tms.xml',
          outputBounds=[df['st_xmin'][i],df['st_ymin'][i],df['st_xmax'][i],df['st_ymax'][i]],
          dstSRS='EPSG:3857')
      else:
        imagery_filename = '/data/shangrui/c/poor/%s.tif'%(df['tower'][i])
        gdal.Warp(
          imagery_filename,
          'google_tms.xml',
          outputBounds=[df['st_xmin'][i],df['st_ymin'][i],df['st_xmax'][i],df['st_ymax'][i]],
          dstSRS='EPSG:3857')

      
      

 
