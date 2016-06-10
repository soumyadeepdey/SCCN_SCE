




#include "StandardHeaders.h"



namespace IITkgp_functions {

 

    /**
     * @function calculate_value
     * @param input RGB value of a pixel
     * @brief Calculate value from RGB values
     * @return Value calculated from RGB values of a pixel
     */


    double calculate_value(int R, int G, int B);

    /**
     * @function calculate_minimum
     * @param input three integer value
     * @brief Calculate minimum between 3 integer value
     * @return Minimum integer value among the 3
     */

    int calculate_minimum(int a, int b, int c);

    /**
     * @function calculate_saturation
     * @param input RGB values of a pixel
     * @brief Calculate saturation from RGB values of a pixel
     * @return calculated S value
     */

    double calculate_saturation(int R, int G, int B);

    /**
     * @function calculate_theta
     * @param input RGB values of a pixel
     * @brief Calculate angle in H domain from given RGB values of a pixel
     * @return calculated theta value
     */

    double calculate_theta(int R, int G, int B);

    /**
     * @function calculate_hue
     * @param input RGB values of a pixel
     * @brief Calculate hue from given RGB values of a pixel
     * @return calculated H value
     */


    double calculate_hue(int R, int G, int B);

    /**
     * @function FindHSVImage
     * @param input RGB color Image
     * @brief Calculate HSV image from the Color RGB image
     * @return calculated HSV image
     */


    Mat FindHSVImage(Mat colorimage);

    /**
     * @function FindNormalizeSaturation
     * @param input Saturated data(double)
     * @brief Calculate normalized saturation value from double value
     * @return calculated S value
     */


    int FindNormalizeSaturation(double saturation);

    /**
     * @function FindNormalizeHue
     * @param input Hue data (value)
     * @brief Calculate normalized Hue value from double value
     * @return calculated H value
     */


    int FindNormalizeHue(double hue);

    /**
     * @function FindHSVImage
     * @param input RGB color Image
     * @brief Calculate nOrmalized HSV image (Range 0-255) from the Color RGB image
     * @return calculated normalized HSV image
     */


    Mat FindNormalizedHSVImage(Mat colorimage);



  
}