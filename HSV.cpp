

#include "HSV.h"

using namespace IITkgp_functions;

/*------------------------------------------------------HSV CALCULATION-----------------------------------------------------*/



/**
 * @function calculate_value
 * @param input RGB value of a pixel
 * @brief Calculate value from RGB values
 * @return Value calculated from RGB values of a pixel
 */


double IITkgp_functions::calculate_value(int R, int G, int B)
{
  double val;
  val = R + G + B;
  val = val/3;
  return val;
}


/**
 * @function calculate_minimum
 * @param input three integer value
 * @brief Calculate minimum between 3 integer value
 * @return Minimum integer value among the 3
 */

int IITkgp_functions::calculate_minimum(int a, int b, int c)
{
  if(a > b)
  {
    if(a > c)
      return a;
    else
      return c;
  }
  else
  {
    if(b > c)
      return b;
    else
      return c;
  }
}


/**
 * @function calculate_saturation
 * @param input RGB values of a pixel
 * @brief Calculate saturation from RGB values of a pixel
 * @return calculated S value
 */

double IITkgp_functions::calculate_saturation(int R, int G, int B)
{
  double sat;
  sat = R + G + B;
  sat = 3/sat;
  sat = sat * calculate_minimum(R, G, B);
  sat = 1 - sat;
  return sat;
}


/**
 * @function calculate_theta
 * @param input RGB values of a pixel
 * @brief Calculate angle in H domain from given RGB values of a pixel
 * @return calculated theta value
 */

double IITkgp_functions::calculate_theta(int R, int G, int B)
{
  double numerator,denomenator,theta,costheta;
  
  
  numerator = 0.5 * ( ( R - G ) + ( R - B ) );
  denomenator = sqrt( ( ( R - G ) * ( R - G ) ) + ( ( R - B ) * ( G - B ) ) );
  costheta = numerator/denomenator;
  theta = acos(costheta);
  theta = theta * 180;
  theta = theta/PI;
  return theta;
  
}


/**
 * @function calculate_hue
 * @param input RGB values of a pixel
 * @brief Calculate hue from given RGB values of a pixel
 * @return calculated H value
 */


double IITkgp_functions::calculate_hue(int R, int G, int B)
{
  double hue;
  if(B <= G)
    hue = calculate_theta(R, G, B);
  else
    hue = 360 - calculate_theta(R, G, B);
  return hue;
}


/**
 * @function FindHSVImage
 * @param input RGB color Image
 * @brief Calculate HSV image from the Color RGB image
 * @return calculated HSV image
 */


Mat IITkgp_functions::FindHSVImage(Mat colorimage)
{
  vector<Mat> rgbimage;
  split(colorimage,rgbimage);
  Mat hsv_image_data;
  double hue,sat,val;
  
  hsv_image_data = Mat(rgbimage[0].rows,rgbimage[0].cols,CV_64FC3);
 
  for(int i=0;i<rgbimage[0].rows;i++)
  {
    for(int j=0;j<rgbimage[0].cols;j++)
    {
      hue = calculate_hue(rgbimage[2].data[i*rgbimage[0].cols+j],rgbimage[1].data[i*rgbimage[0].cols+j],rgbimage[0].data[i*rgbimage[0].cols+j]);
      sat = calculate_saturation(rgbimage[2].data[i*rgbimage[0].cols+j],rgbimage[1].data[i*rgbimage[0].cols+j],rgbimage[0].data[i*rgbimage[0].cols+j]);
      val = calculate_value(rgbimage[2].data[i*rgbimage[0].cols+j],rgbimage[1].data[i*rgbimage[0].cols+j],rgbimage[0].data[i*rgbimage[0].cols+j]);
	hsv_image_data.at<Vec3d>(i,j)[0] = hue;
	hsv_image_data.at<Vec3d>(i,j)[1] = sat;
	hsv_image_data.at<Vec3d>(i,j)[2] = val;
    }
  }
  
  return(hsv_image_data);
}



/**
 * @function FindNormalizeSaturation
 * @param input Saturated data(double)
 * @brief Calculate normalized saturation value from double value
 * @return calculated S value
 */


int IITkgp_functions::FindNormalizeSaturation(double saturation)
{
  int NormalizedSat;
  
    NormalizedSat=(int) saturation*255;
}


/**
 * @function FindNormalizeHue
 * @param input Hue data (value)
 * @brief Calculate normalized Hue value from double value
 * @return calculated H value
 */


int IITkgp_functions::FindNormalizeHue(double hue)
{
  int NormalizedHue;
  double temp;
  
    temp = hue/360;
    NormalizedHue=(int) temp*255;

}


/**
 * @function FindHSVImage
 * @param input RGB color Image
 * @brief Calculate nOrmalized HSV image (Range 0-255) from the Color RGB image
 * @return calculated normalized HSV image
 */


Mat IITkgp_functions::FindNormalizedHSVImage(Mat colorimage)
{
  Mat hsv_image,normalized_hsv_image;
  hsv_image = FindHSVImage(colorimage);
  normalized_hsv_image = Mat(hsv_image.rows,hsv_image.cols,CV_8UC3);
  for(int i=0;i<hsv_image.rows*hsv_image.cols;i++)
  {
    normalized_hsv_image.data[i*3+0] = FindNormalizeHue(hsv_image.data[i*3+0]);
    normalized_hsv_image.data[i*3+1] = FindNormalizeSaturation(hsv_image.data[i*3+1]);
    normalized_hsv_image.data[i*3+2] = hsv_image.data[i*3+2];
    
  }
  
}


/*-------------------------------------------------------------------------------------------------------------------------------------------*/