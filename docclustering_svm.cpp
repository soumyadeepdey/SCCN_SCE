 
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include <sys/stat.h>
#include <iostream>
#include<queue>
#include<list>
#include<stack>
#include<vector>
#include<search.h>
#include <time.h>

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"


#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define PI 3.14159265


#include "ap.h"
#include "alglibinternal.h"
#include "alglibmisc.h"
#include "linalg.h"
#include "specialfunctions.h"
#include "statistics.h"

/*
#include "src/ap.cpp"
#include "src/alglibinternal.cpp"
#include "src/alglibmisc.cpp"
#include "src/linalg.cpp"
#include "src/specialfunctions.cpp"
#include "src/statistics.cpp"

*/

using namespace alglib;




#define _NBC_ 0 // normal Bayessian classifier
#define _KNN_ 0 // k nearest neighbors classifier
#define _SVM_ 1 // support vectors machine
#define _DT_  0 // decision tree
#define _BT_  0 // ADA Boost
#define _GBT_ 0 // gradient boosted trees
#define _RF_  0 // random forest
#define _ERT_ 0 // extremely randomized trees
#define _ANN_ 0 // artificial neural networks
#define _EM_  0 // expectation-maximization


/*-------------------------------------------------------- Structure Declarations--------------------------------------------------------*/


struct BeanStucture{
  int BeanNumber;
  int MaxElement;
  int LowerBound;
  int UpperBound;
  int middle; 
};

struct Ray {
        Point2i p;
        Point2i q;
        vector<Point2i> points;
	int dist;
	void CalcEcluiDist()
	{
	  dist =(int) sqrt( ((p.x - q.x)*(p.x-q.x)) + ((p.y-q.y)*(p.y-q.y)) );
	}
};


typedef struct imagestructure
{
	int x;
	int y;
	int label;
	int value;
	int mapped;

}is;


typedef struct connected_component
{
	struct imagestructure im;
	struct connected_component *nxt;
}cc;



typedef struct numberofcc
{
	struct connected_component *start;
	struct connected_component *last;
	float gray_hist[256];
	/*int text_nontext
	 * @value : 0 if text
	 *          1 if non_text
	 */
	int text_nontext;
	/*int label
	 * @value : value of the color label from k-mean
	 */
	int label;
	int number_of_component;
	int flag;
	int xmin;
	int ymin;
	int xmax;
	int ymax;
	int centroidx;
	int centroidy;
	int blk_no;
	int valid_blk_no;
	float histogram[256];
	double mean;
	double var;
	double std_dev;
	double skew;
	void calc_centroid()
	{
	  centroidx =  xmin+((xmax-xmin)/2);
	  centroidy =  ymin+((ymax-ymin)/2);
	}
	void calc_hist()
	{
	  
	  for(int i=0;i<256;i++)
	    histogram[i]=gray_hist[i]/number_of_component;
	}
	
	void calc_mean()
	{
	  mean = 0;
	  for(int i=0;i<256;i++)
	  {
	    mean = mean + (gray_hist[i]*i);
	  }
	  mean = mean/number_of_component;
	}
	
	
	void calc_dev()
	{
	  calc_mean();
	  int temp;
	  std_dev = 0.0;
	  var = 0.0;
	  skew = 0.0;
	  for(int i=0;i<256;i++)
	  {
	    for(int j=0;j<gray_hist[i];j++)
	    {
	      temp = i - mean;
	      var = var + (temp * temp);
	      skew = skew + (temp * temp *temp);
	    }
	  }
	  
	  var = var / number_of_component;
	  std_dev = sqrt(var);
	  
	}
	
	void calc_distribution()
	{
	  calc_hist();
	  calc_mean();
	  calc_dev();
	}
	
}nocc;

typedef struct clusterinfo
{
  int cluster_label; 
  vector<Point2i> p;
  vector<int> blocks;
  vector<int> block_labels;
}cluster;


typedef struct TrainDataClass
{
  Mat TrainData;
  Mat TrainClass;
  vector<int> ClassNumber;
}TDC;



const int vecsize = 4;

typedef Vec<float, vecsize> FeatureVec;

//typedef Vec<float, 10> Vec10f;


class ProcessingBlock
{
  public: 
    Mat ColorBlock;
    Mat BinaryBlock;
    vector<Point> BlockContour; 
    bool valid;
    
    
  
    vector<Point> GetApproxPoly()
    {
      SetApproxpoly();
      return(BlockPoly);
    }
  
    Rect GetBoundRect()
    {
      SetBoundRect();
      return(BlockRect);
    }
  
    RotatedRect GetBoundRotatedRect()
    {
      SetRotatedRect();
      return(BlockRotatedRect);
    }
    
    void Setfeature(FeatureVec elem)
    {
      if(valid)
      {
	feature = elem;
      }
      else
      {
	printf("This Block is not valid .... You can not set any feature\n");
      }
    }
    
    FeatureVec Getfeature()
    {
      if(valid)
	return(feature);
      else
	printf("This Block is not valid .... There is no feature for this block\n");
    }
    
    void SetClassLabel(int label)
    {
      ClassLabel = label;
      IsLabeled = true;
    }
    
    int GetClassLabel()
    {
      if(IsLabeled)	
	return(ClassLabel);
      else
      {
	printf("No Groundtruth Label is not set\n");
	exit(0);;
      }
    }
    
    int GetPredictedLabel()
    {
      if(IsPredicted)
	return(PredictedLabel);
      else
      {
	printf("No Label is set\n");
	exit(0);;
      }
    }
    
     void SetPredictedLabel(int label)
    {
      PredictedLabel = label;
      IsPredicted = true;
    }
    
     void initialize()
    {
      IsPredicted = false;
      IsPredicted = false;
      valid = false;
    }
  
  private:
    vector<Point> BlockPoly;
    Rect BlockRect;
    RotatedRect BlockRotatedRect;
    FeatureVec feature;
    int ClassLabel;
    int PredictedLabel;
    bool IsPredicted;
    bool IsLabeled;
    
   
    
    void SetApproxpoly()
    {
      approxPolyDP( Mat(BlockContour), BlockPoly, 3, true );
    }
  
    void SetBoundRect()
    {
      if(BlockPoly.empty())
      {
	SetApproxpoly();
	BlockRect = boundingRect( Mat(BlockPoly) );
      }
      else
	BlockRect = boundingRect( Mat(BlockPoly) );
      
    }
  
    void SetRotatedRect()
    {
      
      if(BlockPoly.empty())
      {
	SetApproxpoly();
	BlockRotatedRect = minAreaRect( Mat( BlockPoly ) );
      }
      else
	BlockRotatedRect = minAreaRect( Mat( BlockPoly ) );
      
      
    }
  
  
  
};


class ConfusionMatrix
{
public:
  int tp;
  int fp;
  int tn;
  int fn;
  vector<int> multiclassCM;
  
  void initializeMulticlass(int NoOfClass)
  {
    multiclassCM.resize(NoOfClass,0);
  }
  
  float GetAccuracy()
  {
    SetAccuracy();
    return(accuracy);
  }
  float GetPrecesion()
  {
    SetPrecession();
    return(precesion);
  }
  float GetRecall()
  {
    SetRecall();
    return(recall);
  }
  
  void initialize()
  {
    tp = 0;
    fp = 0;
    tn = 0;
    fn = 0;
    accuracy = 0.0;
    precesion = 0.0;
    recall = 0.0;
  }
  
private:
  float accuracy;
  float precesion;
  float recall;
  
  void SetAccuracy()
  {
    accuracy = ((tp+tn)*1.0)/((tp+fp+tn+fn)*1.0);
  }
  
  void SetPrecession()
  {
    precesion = (tp*1.0)/(tp+fp);
  }
  
  void SetRecall()
  {
    recall = (tp*1.0)/(tp+fn);
  }
  
};


class MultiClassPerformanceMetrics
{
public:
  void initialize(vector<ConfusionMatrix> X)
  {
    CM = X;
    AvgAcc = 0.0;
    ErrorRate = 0.0;
    PrecesionM = 0.0;
    PrecesionMu = 0.0;
    RecallM = 0.0;
    RecallMu = 0.0;
    FScoreM = 0.0;
    FScoreMu = 0.0;
    size = CM.size();
  }
  
  void Clear()
  {
    CM.clear();
    AvgAcc = 0.0;
    ErrorRate = 0.0;
    PrecesionM = 0.0;
    PrecesionMu = 0.0;
    RecallM = 0.0;
    RecallMu = 0.0;
    FScoreM = 0.0;
    FScoreMu = 0.0;
    size = 0;
  }
  
  float GetAverageAccuracy()
  {
    AvgAcc = 0.0;
    size = CM.size();
    for(int i=0;i<CM.size();i++)
    {
      AvgAcc = AvgAcc + GetAccuracy(CM[i]);
    }
    AvgAcc = AvgAcc/(size * 1.0);
    return(AvgAcc);
  }
  
  float GetErrorRate()
  {
    ErrorRate = 0.0;
    size = CM.size();
    for(int i=0;i<CM.size();i++)
    {
      ErrorRate = ErrorRate + GetPerClassError(CM[i]);
    }
    ErrorRate = ErrorRate/(size * 1.0);
    return(ErrorRate);
  }
  
  float GetPrecesionMu()
  {
    float neu = 0.0;
    float deno = 0.0;
    for(int i=0;i<CM.size();i++)
    {
      neu = (neu +CM[i].tp) * 1.0;
      deno = (deno + CM[i].tp + CM[i].fp) * 1.0;
    }
    PrecesionMu = neu/deno;
    return(PrecesionMu);
  }
  
  float GetRecallMu()
  {
    float neu = 0.0;
    float deno = 0.0;
    for(int i=0;i<CM.size();i++)
    {
      neu = (neu + CM[i].tp) * 1.0;
      deno = (deno + CM[i].tp + CM[i].fn)*1.0;
    }
    RecallMu = neu/deno;
    return(RecallMu);
  }
  
  float GetFScoreMu(int Beta)
  {
    float pre = GetPrecesionMu();
    float re = GetRecallMu();
    float neu = ((Beta * Beta) + 1) * pre * re;
    float deno = ((Beta * Beta) * pre) + re;
    
    FScoreMu = neu/deno;
    return(FScoreMu);
  }
  
  float GetPrecesionM()
  {
    PrecesionM = 0.0;
    size = CM.size();
    for(int i=0;i<CM.size();i++)
    {
      PrecesionM = PrecesionM + GetPrecesion(CM[i]);
    }
    PrecesionM = PrecesionM/(size * 1.0);
    return(PrecesionM);
  }
  
  float GetRecallM()
  {
    RecallM = 0.0;
    size = CM.size();
    for(int i=0;i<CM.size();i++)
    {
      RecallM = RecallM + GetRecall(CM[i]);
    }
    RecallM = RecallM/(size * 1.0);
    return(RecallM);
  }
  
  float GetFScoreM(int Beta)
  {
    float pre = GetPrecesionM();
    float re = GetRecallM();
    float neu = ((Beta * Beta) + 1) * pre * re;
    float deno = ((Beta * Beta) * pre) + re;
    
    FScoreM = neu/deno;
    return(FScoreM);
  }
 
  
private:
  vector<ConfusionMatrix> CM;
  float AvgAcc;
  float ErrorRate;
  float PrecesionMu;
  float RecallMu;
  float FScoreMu;
  float PrecesionM;
  float RecallM;
  float FScoreM;
  int size;
  
  float GetAccuracy(ConfusionMatrix C)
  {
    float acc = 0.0;
    if((C.tp+C.fp+C.tn+C.fn) > 0)
    {
      acc = ((C.tp + C.tn)*1.0)/((C.tp+C.fp+C.tn+C.fn)*1.0);
      return(acc);
    }
    else
    {
      size = size - 1;
      return(acc);
    }
  }
  float GetPerClassError(ConfusionMatrix C)
  {
    float err = 0.0;
    if((C.tp+C.fp+C.tn+C.fn) > 0)
    {
      err = ((C.fp+C.fn)*1.0)/((C.tp+C.fp+C.tn+C.fn)*1.0);
      return(err);
    }
    else
    {
      size = size - 1;
      return(err);
    }
  }
  float GetPrecesion(ConfusionMatrix C)
  {
    if((C.tp + C.fp) > 0)
    {
      float pre = (C.tp*1.0)/((C.tp + C.fp)*1.0);
      return(pre);
    }
    else
    {
      size =  size - 1;
      return(0.0);
    }
  }
  float GetRecall(ConfusionMatrix C)
  {
    if((C.tp + C.fn) > 0)
    {
      float re = (C.tp*1.0)/((C.tp + C.fn)*1.0);
      return(re);
    }
    else
    {
      size = size - 1;
      return(0.0);
    }
  }
    
  
};



/*-----------------------------------------------------------------------------------------------------------------*/

/*------------------------------------------------------- Global variables----------------------------------------------------------------------*/



Mat src, src_gray, binary_dst, output_image, erode_dst, dilate_dst, color_para, para_fill,dst;

//Mat *block;

Mat temp_grey,temp_bin; 
Mat temp_src;

int maximum;

int ncc;
int tncc;// total number of connected component
int ncco;//number of connected component in original image

int max_label_forCC;


int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;


Mat temp,hsv,open_dst,YCrCb;

float h_unitvec,s_unitvec,v_unitvec;

// varible hold all hsv values separately
vector<Mat> hsv_planes;

vector<bool> validblock;


int no_of_foregrnd_pix;

RNG rng(12345);



char *substring;

  
double *hue_image,*sat_image,*val_image;
	 
vector<vector<int> > MultiClassResSCCN;
vector<vector<int> > MultiClassResSCE;
	 
	 
/*--------------------------------------------------------Sorting-----------------------------------------------------------------------------------*/


void swaping(std::vector<int> & data, int i, int j)
{
    int tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
}


/*-------------------------------------------------------Merge Sort----------------------------------------------------------*/



void Merge(std::vector<int> & data, int lowl, int highl, int lowr, int highr);
void MergeSort(std::vector<int> & data, int low, int high)
{
    if (low >= high)
    {
        return;
    }
    
    int mid = low + (high-low)/2;

    MergeSort(data, low, mid);

    MergeSort(data, mid+1, high);

    Merge(data, low, mid, mid+1, high);
}

void Merge(std::vector<int> & data, int lowl, int highl, int lowr, int highr)
{
    int tmp_low = lowl;
    std::vector<int> tmp;
    
    while (lowl <= highl && lowr <= highr)
    {
        if (data[lowl] < data[lowr])
        {
            tmp.push_back(data[lowl++]);
        }
        else if (data[lowr] < data[lowl])
        {
            tmp.push_back(data[lowr++]);
        }
        else
        {
            tmp.push_back(data[lowl++]);
            tmp.push_back(data[lowr++]);
        }
    }

    while (lowl <= highl)
    {
        tmp.push_back(data[lowl++]);
    }

    while (lowr <= highr)
    {
        tmp.push_back(data[lowr++]);
    }

    std::vector<int>::const_iterator iter = tmp.begin();
    
    for(; iter != tmp.end(); ++iter)
    {
        data[tmp_low++] = *iter;
    }
}


/*-------------------------------------------------------Quick Sort----------------------------------------------------------*/


int Partition(std::vector<int> & data, int low, int high);
void QuickSort(std::vector<int> & data, int low, int high)
{
    if (low >= high) return;
    
    int p = Partition(data, low, high);

    QuickSort(data, low, p-1);
    QuickSort(data, p+1, high);
}

int Partition(std::vector<int> & data, int low, int high)
{
    int p = low;
    for (int i = p+1; i <= high; ++i)
    {
        if (data[i] < data[p])
        {
            swaping(data, i, p);
            if (i != p+1)
            {
                swaping(data, i, p+1);
            }
            p = p + 1;
        }
    }

    return p;
}



/*-------------------------------------------------------Bubble Sort----------------------------------------------------------*/



//useful for small lists, and for large lists where data is
//already sorted
void BubbleSort(std::vector<int> & data)
{
    int length = data.size();

    for (int i = 0; i < length; ++i)
    {
        bool swapped = false;
        for (int j = 0; j < length - (i+1); ++j)
        {
            if (data[j] > data[j+1])
            {
                swaping(data, j, j+1);
                swapped = true;
            }
        }
        
        if (!swapped) break;
    }
}

/*-------------------------------------------------------Selection Sort----------------------------------------------------------*/



//useful for small lists and where swapping is expensive
// does at most n swaps
void SelectionSort(std::vector<int> & data)
{
    int length = data.size();

    for (int i = 0; i < length; ++i)
    {
        int min = i;
        for (int j = i+1; j < length; ++j)
        {
            if (data[j] < data[min])
            {
                min = j;
            }
        }

        if (min != i)
        {
            swaping(data, i, min);
        }
    }
}


/*-------------------------------------------------------Insertion Sort----------------------------------------------------------*/



//useful for small and mostly sorted lists
//expensive to move array elements
void InsertionSort(std::vector<int> & data)
{
    int length = data.size();

    for (int i = 1; i < length; ++i)
    {
        bool inplace = true;
        int j = 0;
        for (; j < i; ++j)
        {
            if (data[i] < data[j])
            {
                inplace = false;
                break;
            }
        }

        if (!inplace)
        {
            int save = data[i];
            for (int k = i; k > j; --k)
            {
                data[k] = data[k-1];
            }
            data[j] = save;
        }
    }
}



/*-------------------------------------------------------N Choose R----------------------------------------------------------*/

 
bool calcFactorial (int n, int* nfact)
{
    *nfact = 1;
 
    while(n > 0)
    {
           *nfact = *nfact * n;
           n--;   
    }
 
    if(*nfact < 0x7fffffff)
    {
        return true;
    }
    else
    {
        return false;
    }
}
 
/*
Combination means C(n,r) = n!/( r! * (n-r)! )
where C(n,r) is the number of r-element subsets of an n-element set.
Better formula derived from above is:
          n ( n-1 ) ( n-2 ) ... ( n-r+1 )
 C(n,r) = -------------------------------
          r ( r-1 ) ( r-2 ) ... (3)(2)(1)
 
 
  Return True if calculation is successful. False if
  Overflow occurs.
*/
 
bool calcCNR(int n, int r, int *cnr)
{
#define min(n,r)  (((n) < (r)) ? (n) : (r));
    int answer = 1;
    int multiplier = n;
    int divisor = 1;
    int k;
    if(n == r)
    {
      *cnr = answer;
      return true;
      
    }
    else
    {
 
      k = min(r, n - r);
    // printf("n=%d, r=%d, k=%d\n", n, r, k);
  
      while (divisor <= k) {
	  answer = ((answer * multiplier) / divisor);
	  *cnr = answer;
	// printf("Intermediate answer=%d\n", answer);
	  multiplier--;
	  divisor++;
      }
      if(*cnr < 0x7fffffff)
      {
	  return true;
      }
      else
      {
	  return false;
      }
    }
}




/*-------------------------------------------------cut string upto( .)-------------------------------------------*/

/**
 * @function input_image_name_cut
 * @param : input param: input-name to be cut upto '.'
 * @return : input name upto '.' 
 *
 */


char* input_image_name_cut(char *s) 
{
                 
                     int i,j; 
		     
		     char *substring;
		     
		     substring = (char *)malloc(2001 * sizeof(char));
              
                 for(i=0; i <= strlen(s)-1; i++)
                      {
			
                       if (s[i]!='.' )
		        substring[i] = s[i];
		       else
			 break;
                       }
                       substring[i] = '\0';
                 
                     printf(" %s\n", substring);
		 
		 return(substring);
		     
		     
                      
      }



/*-------------------------------------------------------------------------------------------------------------------------------------------*/



/*-------------------------------------------------MAKE DIRECTORY FUNCTION-------------------------------------------*/
/**
 * @function makedir
 * @param input a character string
 * @brief it create a directry of given character string
 */
void makedir(char *name)
{
	int status;
	status=mkdir(name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}



/*-------------------------------------------------------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------------------------------------------------------------------------*/


/**
 * @function CreateNameIntoFolder
 * @param  input Foldername, desired name 
 * @return : name within the desired folder
 *
 */

char* CreateNameIntoFolder(char *foldername, char *desiredname )
{
  char *name,*output,*tempname, *tempname1;
  output = (char *) malloc ( 2001 * sizeof(char));
  if(output == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  strcpy(output,foldername);
 
  tempname = (char *) malloc ( 2001 * sizeof(char));
  if(tempname == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  tempname = "/";
  strcat(output,tempname);
  
  tempname1 = (char *) malloc ( 2001 * sizeof(char));
  if(tempname1 == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  strcpy(tempname1,output);
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  strcpy(name,tempname1);
  
  strcat(name,desiredname);
  
  return(name);
  
  
}


/*-------------------------------------------------------------------------------------------------------------------------------------------*/

 



/*-------------------------------------------------------------------------------------------------------------------------------------------*/






/**
 * @function validate
 * @param : input param: co-ordinate position(i,j) and maximum-limit(t) row, col
 * @brief : co-ordinate position(i,j) to be check whether it is within given row and col
 * @return : 1 if it belong to particular region
 *           0 if not belong within that particular row and col
 */


int validate(int i, int j, int row, int col)
{
  
  if(i<0 || i>=row || j<0 || j>=col)
    return 0;
  else
    return 1;
}


/*-------------------------------------------------------------------CONNECTED COMPONENT--------------------------------------------------------------------------------*/


/**
 * @function labelingdfs
 * @param : input param: input image structure(of (IS(datatype))) input co-ordinate(m,n) to belabeled and image dimention(row,col)
 * @brief : labeling based on DFS algorithm
 * @return : image structure(of (IS(datatype)))
 */



is ** labelingdfs(is **tempstructuredimage,int *m, int *n, int temp_row, int temp_col)
{
  
 // printf("hello in DFS and total no of window is %d %d\n",temp_row,temp_col);
  int i,j,l,k,p,q,t;
	i=*m;
	j=*n;
	//printf("iteration \n");
	for(l=i-1;l<=i+1;l++)
	{
	  for(k=j-1;k<=j+1;k++)
	  {
	    if(validate(l,k,temp_row,temp_col))
	    {
	      if(tempstructuredimage[l][k].label==0 && tempstructuredimage[l][k].value==0)
	      {
		tempstructuredimage[l][k].label=tempstructuredimage[i][j].label;
		tempstructuredimage = labelingdfs(tempstructuredimage,&l,&k,temp_row,temp_col);
	      }
	    }
	  }
	}
	
	
	return(tempstructuredimage);
}



// LABELING WITH DFS


/**
 * @function cclabeling
 * @param : input param: input image structure(of (IS(datatype))) and image dimention(row,col)
 * @brief : labeling of ConnectedComponent based on DFS algorithm
 * @return : image structure(of (IS(datatype)))
 */


is** cclabeling(is **tempstructuredimage,int temp_row, int temp_col)
{
	int label=1;
	int i,j,k,l;
        
	for(i=0;i<temp_row;i++)
	{
		for(j=0;j<temp_col;j++)
		{
	
			if(tempstructuredimage[i][j].label==0&&tempstructuredimage[i][j].value==0)
			{
				printf("finding connected_component number %d\n",label);
				tempstructuredimage[i][j].label=label;
				tempstructuredimage = labelingdfs(tempstructuredimage,&i,&j,temp_row,temp_col);
				label=label+1;
			}
			
			
		}
	}
	
	max_label_forCC = label;
	return(tempstructuredimage);
	
}


/**
 * @function FindConnectedComponent
 * @param : input param: input image(binary) Mat data-type
 * @brief : Find connected component from the input binary image
 * @return : array of connected components (nocc(data-type))
 */


nocc* FindConnectedComponent(Mat image)
{
	int row,col;
	int i,j,label=1,*mapping,k;
	ncc=0;
	
	max_label_forCC = 0;
	
	is *istemp;

	row = image.rows;
	col = image.cols;

	is **tempstructuredimage;


	tempstructuredimage=(is **)malloc(row * sizeof(is *));
	if(tempstructuredimage == NULL)
	{
	  printf("can not allocate space\n");
	  exit(0);
	}
	for(i=0;i<row;i++)
	{
		tempstructuredimage[i]=(is *)malloc(col * sizeof(is));
		if(tempstructuredimage[i] == NULL)
		{
		  printf("can not allocate space\n");
		  exit(0);
		}
	}
	
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			tempstructuredimage[i][j].x=i;
			tempstructuredimage[i][j].y=j;
			tempstructuredimage[i][j].label=0;
			tempstructuredimage[i][j].value=image.data[i*col+j];
		}
	}



// LABELING BY DFS
	printf("before labeling\n");
	
	tempstructuredimage = cclabeling(tempstructuredimage,row,col);
	
	printf("after labeling\n");
	int noccbyla;
	noccbyla = max_label_forCC;

// LABELING IS PERFECT

	mapping=NULL;
	k=0;
	int *tmp,count=1;
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			if(tempstructuredimage[i][j].label!=0)
			{
			 
				if(mapping!=NULL)
				{
					for(k=0;k<count-1;k++)
					{
						if(mapping[k]==tempstructuredimage[i][j].label)
						{
							tempstructuredimage[i][j].mapped=k;
							break;
						}		
					}
					if(k==count-1)
					{
						tmp=(int*)realloc(mapping,count*sizeof(int));
						if(tmp!=NULL)
						{
							mapping=tmp;
							mapping[count-1]=tempstructuredimage[i][j].label;
							tempstructuredimage[i][j].mapped=(count-1);
							count++;
						}
						else
						{
							printf("\nERROR IN REALLOCATING MAPPING ARREY\n");
							exit(1);
						}
					}// end of k==count
				}// end of if mapping !=null
				else
				{
				  
					tmp=(int*)realloc(mapping,count*sizeof(int));
					mapping=tmp;
					mapping[count-1]=tempstructuredimage[i][j].label;
					tempstructuredimage[i][j].mapped=(count-1);
					count++;
					
				}
				
			}// end of tempstructuredimage[i][j].label!=0
		}
	}// end of image

	

// MAPPING IS PERFECR TILL NOW

printf("MAPPING IS PERFECR TILL NOW\n");

	tncc=count-1;
	
	ncco = tncc;

// CREATING ARREY OF STRUCTURE POINTER  and help them to uniquely mapped

	cc *cctemp,*ccstart=NULL,*temp1;
	
	nocc *component;
	
	component=(nocc *)malloc((count-1)* sizeof(nocc));
	
	if(component == NULL)
	{
	  printf("memory can not be allocated \n");
	  exit (0);
	}

	for(i=0;i<(count-1);i++)
	{
		component[i].start=NULL;
		component[i].number_of_component=0;
		component[i].last=NULL;
		for(j=0;j<256;j++)
		  component[i].gray_hist[j]=0.0;
	}
	

	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			if(tempstructuredimage[i][j].label!=0)
			{
				
				if(component[tempstructuredimage[i][j].mapped].start==NULL)
				{
					
					if(tempstructuredimage[i][j].mapped<0||tempstructuredimage[i][j].mapped>=count-1)
						{
							printf("error\n");
							printf("%d\t%d\t%d\t%d",tempstructuredimage[i][j].mapped,tempstructuredimage[i][j].x,tempstructuredimage[i][j].y,tempstructuredimage[i][j].label);
							exit(1);
						}
				
					
					cctemp=(cc *)malloc(sizeof(cc));
					if(cctemp == NULL)
					{
					  printf("memory can not be allocated \n");
					  exit (0);
					}
					cctemp->im.x=i;
					cctemp->im.y=j;
					cctemp->im.label=tempstructuredimage[i][j].label;
					cctemp->im.mapped=tempstructuredimage[i][j].mapped;
					cctemp->im.value=src_gray.data[i*col+j];
					cctemp->nxt=NULL;
					ccstart=(cc *)malloc(sizeof(cc));
					ccstart=cctemp;
					component[tempstructuredimage[i][j].mapped].start=cctemp;
					component[tempstructuredimage[i][j].mapped].last=cctemp;
					component[tempstructuredimage[i][j].mapped].number_of_component=1;
					component[tempstructuredimage[i][j].mapped].xmin=i;
					component[tempstructuredimage[i][j].mapped].ymin=j;
					component[tempstructuredimage[i][j].mapped].xmax=i;
					component[tempstructuredimage[i][j].mapped].ymax=j;
					component[tempstructuredimage[i][j].mapped].gray_hist[src_gray.data[i*col+j]]=component[tempstructuredimage[i][j].mapped].gray_hist[src_gray.data[i*col+j]]+1;
					
				}//end of if  i.e. first component of the connected component
				else
				{
					
					cctemp=(cc *)malloc(sizeof(cc));
					if(cctemp == NULL)
					{
					  printf("memory can not be allocated \n");
					  exit (0);
					}
					cctemp->im.x=i;
					cctemp->im.y=j;
					cctemp->im.label=tempstructuredimage[i][j].label;
					cctemp->im.mapped=tempstructuredimage[i][j].mapped;
					cctemp->im.value=src_gray.data[i*col+j];
					cctemp->nxt=NULL;
					if(component[tempstructuredimage[i][j].mapped].last->nxt==NULL)
						component[tempstructuredimage[i][j].mapped].last->nxt=cctemp;
					else
						printf("ERROR\n");
					component[tempstructuredimage[i][j].mapped].last=cctemp;
					component[tempstructuredimage[i][j].mapped].number_of_component=(component[tempstructuredimage[i][j].mapped].number_of_component)+1;
					if(component[tempstructuredimage[i][j].mapped].xmin>i)
						component[tempstructuredimage[i][j].mapped].xmin=i;
					if(component[tempstructuredimage[i][j].mapped].ymin>j)
						component[tempstructuredimage[i][j].mapped].ymin=j;
					if(component[tempstructuredimage[i][j].mapped].xmax<i)
						component[tempstructuredimage[i][j].mapped].xmax=i;
					if(component[tempstructuredimage[i][j].mapped].ymax<j)
						component[tempstructuredimage[i][j].mapped].ymax=j;
					
					component[tempstructuredimage[i][j].mapped].gray_hist[src_gray.data[i*col+j]]=component[tempstructuredimage[i][j].mapped].gray_hist[src_gray.data[i*col+j]]+1;

				}// end of else i.e. not 1st component of connected component
			}// end of if label

			
	
		}// end of j

	
	
	}// end of i
	printf("CC done\n");
	free(mapping);
	
	return(component);

}





/*------------------------------------------------------BINARIZATION-------------------------------------------------------------------*/



// parameters for binarization

int binary_threshold_value = 211;

  /**
   * @param :thereshold_type
   * 
   * 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */
int threshold_type = 0;
int const maximum_value = 255;
int const maximum_type = 4;
int const maximum_BINARY_value = 255;
int const blockSize=251;
//int const blockSize=101;
Mat TempGray,TempBinary;



void BinaryThreshold( int, void* )
{
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */

  threshold( TempGray, TempBinary,  binary_threshold_value, maximum_BINARY_value,threshold_type );
  imshow("BinaryThresholding",TempBinary);
}


/**
 * @function binarization
 * @param input an image in Mat format and type for binarization
 * @brief type = 1 for adaptive
 * @brief type = 2 for otsu
 * @brief type = 3 for Normal Threshold by GUI
 * @brief type = 4 for Normal Threshold by fixed value
 * @return Return a binary image of Mat data-type 
  */

Mat binarization(Mat image, int type)
{
/**
 * @param type 
 * type = 1 for adaptive;
 * type = 2 for Otsu
 * type = 3 for Normal Threshold by GUI   
**/
       
	// Convert the image to Gray
  	
  	Mat gray,binary;
	threshold_type = 0;
	
	cvtColor(image, gray, CV_BGR2GRAY);
	
	if(type == 1)
	{
	  adaptiveThreshold(  gray, binary, maximum_BINARY_value, ADAPTIVE_THRESH_GAUSSIAN_C,  threshold_type,  blockSize, 0);
	  return (binary);
	}
	
	// Otsu Thresholding
	if(type == 2)
	{
	  double val = threshold( gray, binary, 100, maximum_BINARY_value, cv::THRESH_OTSU | cv::THRESH_BINARY);
	  printf("threshold value is %lf\n",val);
	  return (binary);
	}
	
	//GUI Threshold
	if(type == 3)
	{
	  gray.copyTo(TempGray);
	  /// Create a window to display results
	    namedWindow( "BinaryThresholding", CV_WINDOW_KEEPRATIO );
	    
	    createTrackbar( "Value",
			    "BinaryThresholding", & binary_threshold_value,
			    maximum_BINARY_value, BinaryThreshold );

	    /// Call the function to initialize
	    BinaryThreshold( 0, 0 );
	    waitKey(0);
	    printf("threshold value is %d\n",binary_threshold_value);
	    destroyWindow("BinaryThresholding");
	    TempBinary.copyTo(binary);
	    return (binary);
	}
	
	// Fixed Threshold
	if(type == 4)
	{
	  binary_threshold_value = 211;
	  threshold( gray, binary,  binary_threshold_value, maximum_BINARY_value,threshold_type );
	  return (binary);
	}
	
	
	  
	

}



/*------------------------------------------------------------------------------------------------------------------------------------------------*/


/**
 * @function foreground_masked_image
 * @param input a color image in Mat format and it's corresponding binary image
 * @brief convert an input image into a uniform background image
 * @brief masked the foreground pixels and make the background pixel uniform
 * @return Return a uniform background image of Mat data-type 
 */


Mat foreground_masked_image(Mat ColorImage, Mat binary)
{
  Mat uniform;
  
  //binary = binarization(ColorImage,2);
  ColorImage.copyTo(uniform);
  int row = ColorImage.rows;
  int col = ColorImage.cols;
  
  for(int i =0;i<row;i++)
  {
    for(int j=0;j<col;j++)
    {
      if(binary.data[i*col+j] == 255)
      {
	for(int k=0;k<3;k++)
	  uniform.data[(i*col+j)*3+k]=255;
      }
    }
  }
  
  return(uniform);
  
}


/*------------------------------------------------------------------------------------------------------------------------------------------------*/


/**
 * @function NumberofForegroundPixel
 * @param input a binary image in Mat format
 * @brief It count number of foreground pixel in the given image
 * @return Return a integer which gives the count of number of foreground pixel 
 */


int NumberofForegroundPixel(Mat image)
{
 // Mat binary;
  int row = image.rows;
  int col = image.cols;
  int pixel_count=0;
 // binary = binarization(image,2);
  
  for(int i =0;i<row;i++)
  {
    for(int j=0;j<col;j++)
    {
      if(image.data[i*col+j] == 0)
	pixel_count = pixel_count + 1;
    }
  }
  
  return(pixel_count);
}


/*------------------------------------------------------------------------------------------------------------------------------------------------*/


/*----------------------------------------------MORPHOLOGICAL OPERATIONS----------------------------------------------------------------------*/



/*-------------------------------------------------------EROTION WITH 4 NEIGHBOURHOOD-------------------------------------------------------------*/


/**
 * @function erosion
 * @param input an image(binary) in Mat format
 * @brief it erode an image with a square mask of 3x3
 * @return Return eroded image of Mat data-type
 */

Mat erosion(Mat image)
{
	int row = image.rows;
	int col = image.cols;
	int i,j;
	Mat tempimage;
	image.copyTo(tempimage);
	for(i=0;i<row;i++)
	{
	  for(j=0;j<col;j++)
	    tempimage.data[i*col+j] = 255;
	}
	
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			if(image.data[i*col+j]==0)
			{
				if(i-1<0||i+1>=row||j-1<0||j+1>=col)
					tempimage.data[i*col+j]=255;
				else if(image.data[(i-1)*col+j]==0&&image.data[(i+1)*col+j]==0&&image.data[i*col+(j-1)]==0&&image.data[i*col+(j+1)]==0)
					tempimage.data[i*col+j]=0;
				else
					tempimage.data[i*col+j]=255;
			}
			else
				tempimage.data[i*col+j]=255;
		}
	}

	return (tempimage);
	
		
}

/*-------------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------- BOUNDARY EXTRACTION--------------------------------------------------*/


/**
 * @function boundaryextraction
 * @param input an image(binary) in Mat format
 * @brief it find the boundary of the input image 
 * @return Return boundary of input image(binary in nature)
 */

Mat boundaryextraction(Mat image)
{
	
	int i,j,k;

	Mat erodedimage;
	Mat extractedimage;
	
	image.copyTo(erodedimage);
	image.copyTo(extractedimage);
	int row,col;
	
	row = image.rows;
	col = image.cols;
	
	for(i=0;i<row;i++)
	{
	  for(j=0;j<col;j++)
	  {
	    erodedimage.data[i*col+j] = 255;
	    extractedimage.data[i*col+j] = 255;
	  }
	}
	
	erodedimage=erosion(image);
	
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
		{
			if(image.data[i*col+j]==erodedimage.data[i*col+j])
				extractedimage.data[i*col+j]=255;
			else
				extractedimage.data[i*col+j]=0;
		}
	}

	return(extractedimage);
	
	
}




/**  @function Erosion  
 * @param input 
 * element type
 * 0: kernel = Rectangle
 * 1: kernel = CROSS
 * 2: kernel = ELLIPSE
 * @param input erosion Size(n) : Create a kernel or window of 2n+1
 * @param input an image in Mat format(image).
 * @brief it find Eroded Image of the input image with given kernel type and size 
 * @return Return Eroded image of input image
 */
Mat Erosion( int erosion_elem, int erosion_size, Mat image)
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
  Mat ErodedImage;
  /// Apply the erosion operation
  erode( image, ErodedImage, element );
  return(ErodedImage);
  
}


/**  @function Dilation  
 * @param input 
 * element type
 * 0: kernel = Rectangle
 * 1: kernel = CROSS
 * 2: kernel = ELLIPSE
 * @param input Dilation Size(n) : Create a kernel or window of 2n+1
 * @param input an image in Mat format(image).
 * @brief it find Dilated Image of the input image with given kernel type and size 
 * @return Return Dilateded image of input image
 */
Mat Dilation( int dilation_elem, int dilation_size, Mat image )
{
  Mat DilatedImage;
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( image, DilatedImage, element );
  return(DilatedImage);
  
}


/**  @function Open  
 * @param input 
 * element type   
 * 0: kernel = Rectangle
 * 1: kernel = CROSS
 * 2: kernel = ELLIPSE
 * @param input element Size(n) : Create a kernel or window of 2n+1
 * @param input an image in Mat format(image).
 * @brief it find Open Image of the input image with given kernel type and size 
 * @return Return Open image of input image
 */
Mat Open(int open_elem, int open_size, Mat image)
{
  Mat OpenImage;
  
  int open_type;
  if( open_elem == 0 ){ open_type = MORPH_RECT; }
  else if( open_elem == 1 ){ open_type = MORPH_CROSS; }
  else if( open_elem == 2) { open_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( open_type,
                                       Size( 2*open_size + 1, 2*open_size+1 ),
                                       Point( open_size, open_size ) );
  Mat ErodedImage;
  erode(image, ErodedImage, element);
  dilate(ErodedImage, OpenImage, element);
  //ErodedImage = Erosion(open_elem,open_size, image);
  //OpenImage = Dilation(open_elem,open_size, ErodedImage);
  return(OpenImage);
}


/**  @function Close  
 * @param input  
 * element type   
 * 0: kernel = Rectangle
 * 1: kernel = CROSS
 * 2: kernel = ELLIPSE
 * @param input element Size(n) : Create a kernel or window of 2n+1
 * @param input an image in Mat format(image).
 * @brief it find Close Image of the input image with given kernel type and size 
 * @return Return Close image of input image(binary in nature)
 */
Mat Close(int close_elem, int close_size, Mat image)
{
  Mat CloseImage;
  int close_type;
  if( close_elem == 0 ){ close_type = MORPH_RECT; }
  else if( close_elem == 1 ){ close_type = MORPH_CROSS; }
  else if( close_elem == 2) { close_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( close_type,
                                       Size( 2*close_size + 1, 2*close_size+1 ),
                                       Point( close_size, close_size ) );
  Mat DilatedImage;
  dilate( image, DilatedImage, element );
  erode(DilatedImage, CloseImage, element);
 // DilatedImage = Dilation(close_elem, close_size, image);
 // CloseImage = Erosion(close_elem, close_size, DilatedImage);
  return(CloseImage);
}


/*-------------------------------------------------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------------*/


/**
 * @Function: PointRectangleTest
 * @brief : Take 1 rectangle and a Point as input 
 * @brief : Test whether the Given Point is inside the Given Rectangle or Inside
 * @return : 	0: Point is Outside of Rectangle
 * 		1: Point is inside the given Rectangle
 * */

int PointRectangleTest(Rect GivenRect, Point GivenPoint)
{
  Point tl,br;
  tl = GivenRect.tl();
  br = GivenRect.br();
  int flag;
  /*
  if((GivenPoint.x>=tl.x && GivenPoint.x<=br.x) && (GivenPoint.y<=tl.y && GivenPoint.y>=br.y))
  {
    flag = 1;
    printf("point inside\n");
    return(flag);
  }
  */
  if((GivenPoint.x>=tl.x && GivenPoint.x<=br.x) && (GivenPoint.y>=tl.y && GivenPoint.y<=br.y))
  {
    flag = 1;
    //printf("point inside\n");
    return(flag);
  }
  else
  {
    flag = 0;
    return(flag);
  } 
}

/**
 * @Function: FindOverlappingRectangles
 * @brief : Take 2 rectangle as an input 
 * @return : 	0: Rect 1 and Rect 2 are disjoint
 * 		1: Rect 1 is inside Rect 2
 * 	    	2: Rect 2 is inside Rect 1
 * 	    	3: Rect 1 and 2 are partly overlapped
 * 		
 * 
 * */

int FindOverlappingRectangles(Rect first, Rect second)
{

 Point tl1,tl2,br1,br2;
 
 int flag;

 tl1 = first.tl();
 tl2 = second.tl();
 br1 = first.br();
 br2 = second.br();
 
 if(PointRectangleTest(first,tl2) == 0 && PointRectangleTest(first,br2) == 0)
 {
   flag = 0;
   //return (flag);
 }
 
 if(PointRectangleTest(first,tl2) == 1 || PointRectangleTest(first,br2) ==1 || PointRectangleTest(second,tl1) == 1 || PointRectangleTest(second,br1) == 1)
 {
   flag = 3;
   ///return (flag);
 }
 if(PointRectangleTest(first,tl2) == 1 && PointRectangleTest(first,br2) == 1)
 {
   flag = 2;
   //return (flag);
 }
 if(PointRectangleTest(second,tl1) == 1 && PointRectangleTest(second,br1) == 1)
 {
   flag = 1;
   //return (flag);
 }
 return (flag);
 
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/


/*---------------------------------------------------------SMOOTHING OPERATIONS----------------------------------------------------------------*/


/*----------------------------------------------------------HORIZONTAL SMOOTHING----------------------------------------------------------------*/


/**
 * @function horizontal_smoothing
 * @param input an image(binary) in Mat format and integer value whitespace that need to be smoothen or filled up
 * @brief with this function from a foreground position , next whitespace number of pixel is filled in horizontal direction
 * @brief it produce smoothed image of the input image in horizontal direction by filling up with foreground with whitespace amount
 * @return Return horizontally smoothed image of input image(binary in nature)
 */

Mat horizontal_smoothing(Mat image, int whitespace)
{
	int i,j,k;
	int row = image.rows;
	int col = image.cols;
	
	Mat hsmoothedimage;
	
        image.copyTo(hsmoothedimage);
	
	for(i=0;i<image.rows;i++)
	{
		for(j=0;j<image.cols;j++)
		{			
			if(image.data[(i*image.cols)+j]==0)
			{
				for(k=j+1;k<(j+1+whitespace);k++)
				{
					if(k<image.cols)
					{					
						hsmoothedimage.data[(i*image.cols)+k]=0;
					}
					else 
						break;
					
				}
			}
		}
	}

	
	return(hsmoothedimage);
	
}


/**
 * @function horizontal_gapfilling
 * @param input an image(binary) in Mat format and integer value whitespace that need to be smoothen or filled up
* @brief with this function gap btween two foreground pixel is filled only if the gap between two foreground pixel in horizontal direction has a gap less than or equal to whitespace
 * @brief it produce smoothed image of the input image in horizontal direction by filling up with foreground with whitespace amount
 * @return Return gap-filled image of input image(binary in nature)
 */

Mat horizontal_gapfilling(Mat image, int whitespace)
{
	int i,j,k,l;
	
	Mat hgapfilled;
	
        image.copyTo(hgapfilled);
	
	for(i=0;i<image.rows;i++)
	{
	  for(j=0;j<image.cols;j++)
	  {
	    if(image.data[(i*image.cols)+j]==0)
	    {
	      if( (j+whitespace)<hgapfilled.cols)
	      {
		for(k=j+whitespace;k>=j+1;k--)
		{
		  if(image.data[(i*image.cols)+k]==0)
		  {
		    for(l=k;l>=j+1;l--)
		      hgapfilled.data[(i*hgapfilled.cols)+l]=0;
		    break;
		  }
		}
	      }
	      else
		break;
	    }
	  }
	}
	
	return (hgapfilled);
}

/*---------------------------------------------------------VERTICAL SMOOTHING----------------------------------------------------------------*/


/**
 * @function vertical_smoothing
 * @param input an image(binary) in Mat format and integer value whitespace that need to be smoothen or filled up
 * @brief with this function from a foreground position , next whitespace number of pixel is filled
 * @brief it produce smoothed image of the input image in vertical direction by filling up with foreground with whitespace amount
 * @return Return vertically smoothed image of input image(binary in nature)
 */

Mat vertical_smoothing(Mat image,int whitespace)
{
	int i,j,k;
	
	Mat vsmoothedimage;
	image.copyTo(vsmoothedimage);
	
	for(i=0;i<image.rows;i++)
	{
		for(j=0;j<image.cols;j++)
		{			
			if(image.data[(i*image.cols)+j]==0)
			{
				for(k=i+1;k<(i+1+whitespace);k++)
				{
					if(k<vsmoothedimage.rows)
					{					
						vsmoothedimage.data[(k*vsmoothedimage.cols)+j]=0;
					}
					else 
						break;
					
				}
			}
		}
	}
	
	return (vsmoothedimage);	
}


/**
 * @function vertical_gapfilling
 * @param input an image(binary) in Mat format and integer value whitespace that need to be gap filled or filled up
 * @brief with this function gap btween two foreground pixel is filled only if the gap between two foreground pixel in vertival direction has a gap less than or equal to whitespace
 * @brief it produce gap filled image of the input image in vertical direction by filling up with foreground with whitespace amount
 * @return Return vertically gap-filled image of input image(binary in nature)
 */


Mat vertical_gapfilling(Mat image,int whitespace)
{
	int i,j,k,l;
	
	Mat vgapfilled;
	image.copyTo(vgapfilled);
	for(i=0;i<image.rows;i++)
	{
	  for(j=0;j<image.cols;j++)
	  {
	    if(image.data[(i*image.cols)+j]==0)
	    {
	      if( (i+whitespace)<vgapfilled.rows)
	      {
		for(k=i+whitespace;k>=i+1;k--)
		{
		  if(image.data[(k*image.cols)+j]==0)
		  {
		    for(l=k;l>=i+1;l--)
		      vgapfilled.data[(l*vgapfilled.cols)+j]=0;
		    break;
		  }
		}
	      }
	      else
		break;
	    }
	  }
	}
	
	return (vgapfilled);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/

/*------------------------------------------------------HSV CALCULATION-----------------------------------------------------*/



/**
 * @function calculate_value
 * @param input RGB value of a pixel
 * @brief Calculate value from RGB values
 * @return Value calculated from RGB values of a pixel
 */


double calculate_value(int R, int G, int B)
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

int calculate_minimum(int a, int b, int c)
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

double calculate_saturation(int R, int G, int B)
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

double calculate_theta(int R, int G, int B)
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


double calculate_hue(int R, int G, int B)
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


Mat FindHSVImage(Mat colorimage)
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


int FindNormalizeSaturation(double saturation)
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


int FindNormalizeHue(double hue)
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


Mat FindNormalizedHSVImage(Mat colorimage)
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
/*
 * Normal data Type to Mat dataType
 */

Mat Integer1DArray2Mat(int *data_array, int row, int col)
{
  Mat data;
  data = Mat(row,col,CV_8SC1);
  
  for(int i=0;i<row*col;i++)
    data.data[i]=data_array[i];
  
  return(data);
}


Mat Float1DArray2Mat(float *data_array, int row, int col)
{
  Mat data;
  data = Mat(row,col,CV_16SC1);
  
  for(int i=0;i<row*col;i++)
    data.data[i]=data_array[i];
  
  return(data);
}


Mat double1DArray2Mat(double *data_array, int row, int col)
{
  Mat data;
  data = Mat(row,col,CV_32SC1);
  
  for(int i=0;i<row*col;i++)
    data.data[i]=data_array[i];
  
  return(data);
}


/*-------------------------------------------------------------------------------------------------------------------------------------------*/


/*-----------------------------Statistical function--------------------------------*/


/**
 * @function FindMean
 * @param input Single Channel Mat data
 * @brief Calculate Mean of Given data array
 * @return mean(double) of the given array
 */



double FindMean(Mat data)
{
  data.convertTo(data,CV_64FC1);
  double mean;
  double sum;
  sum = 0.0;
  int data_size;
  data_size = data.rows*data.cols;
  for(int i=0;i<data_size;i++)
    sum = sum + (data.at<double>(0,i));
  mean = sum/data_size;
  
  return(mean);
}


/**
 * @function FindVar
 * @param input Single Channel Mat data
 * @brief Calculate variance of Given data array
 * @return Variance(double) of the given array
 */


double FindVar(Mat data)
{
  data.convertTo(data,CV_64FC1);
  double var,mean;
  mean = FindMean(data);
  double temp;
  double sum=0.0;
  int data_size;
  data_size = data.rows*data.cols;
  for(int i=0;i<data_size;i++)
  {
    temp = data.at<double>(0,i) - mean;
    sum = sum + (temp * temp);
  }
  var = sum/data_size;
  
  return(var);
}


/**
 * @function FindStdDev
 * @param input Single Channel Mat data
 * @brief Calculate Standard Deviation of Given data array
 * @return Standard Deviation(double) of the given array
 */


double FindStdDev(Mat data)
{
  data.convertTo(data,CV_64FC1);
  double std_dev,var;
  var = FindVar(data);
  std_dev = sqrt(var);
  
  return(std_dev);
}


/**
 * @function FindSkew
 * @param input Single Channel Mat data
 * @brief Calculate Skewness of Given data array
 * @return Skewness(double) of the given array
 */


double FindSkew(Mat data)
{
  data.convertTo(data,CV_64FC1);
  double skew;
  double sum = 0.0;
  double mean;
  double temp;
  double std_dev;
  int data_size;
  data_size = data.rows*data.cols;
  mean = FindMean(data);
  
  for(int i=0;i<data_size;i++)
  {
    temp = data.at<double>(0,i) - mean;
    sum = sum + (temp * temp * temp);
  }
  sum = sum / data_size;
  std_dev = FindStdDev(data);
  skew = sum/(std_dev * std_dev *std_dev);
  
}


/**
 * @function FindMinElementPosi
 * @param input Single Channel Mat data and pointer to min element  and its position
 * @brief Calculate Min value of Given data array and its position
 * 
 */



void FindMinElementPosi(Mat data, double *value, int *posi)
{
  data.convertTo(data,CV_64FC1);
  double min_element;
  min_element = data.at<double>(0,0);
  int min_posi;
  int data_size;
  data_size = data.rows*data.cols;
  for(int i=0;i<data.rows;i++)
  {
    for(int j=0;j<data.cols;j++)
    {
      if(data.at<double>(i,j)<=min_element)
      {
	min_element = data.at<double>(i,j);
	min_posi = i*data.cols+j;
      }
    }
  }
  
  *value = min_element;
  *posi = min_posi;
  
}


/**
 * @function FindMaxElement
 * @param input Single Channel Mat data and pointer to max element and pointer to position
 * @brief Calculate Max value of Given data array and its position
 * 
 */



void FindMaxElementPosi(Mat Mdata, double *value, int *posi)
{
  Mat data;
  Mdata.convertTo(data,CV_64FC1);
  double max_element = 0.0;
 // max_element = data.at<double>(0,0);
  int max_posi = 0;
 // int data_size;
 // data_size = data.rows*data.cols;
  for(int i=0;i<data.rows;i++)
  {
    for(int j=0;j<data.cols;j++)
    {
      if(max_element <= data.at<double>(i,j))
      {
	max_element = data.at<double>(i,j);
	max_posi = i*data.cols+j;
      }
    }
  }
 // printf("max value %lf\n",max_element);
  
  *value = max_element;
  *posi = max_posi;
  
}


/**
 * @function FindHistogram
 * @param input Single Channel Mat data
 * @brief Calculate Histogram of the data
 * @return Histogram data of the element
 */



Mat FindHistogram(Mat data)
{
  
 
  Mat HistData;
  double max_elem;
  int max_posi;
 
  FindMaxElementPosi(data,&max_elem,&max_posi);
  
  bool uniform = true; bool accumulate = false;
  int histSize = (int)max_elem;
  //printf("HistSize is %d\t%lf\n",histSize,max_elem);
 // int histSize = 256;
  /// Set the ranges
  float range[] = { 0, histSize } ;
  const float* histRange = { range };
  
  Mat ConvertedData;
  data.convertTo(ConvertedData,CV_8UC1);
  
  /// Compute the histograms:
  calcHist( &ConvertedData, 1, 0, Mat(), HistData, 1, &histSize, &histRange, uniform, accumulate );
  
 
  return(HistData);
  
}

/**
 * @function DrawHistogram
 * @param input Single Channel Mat data
 * @brief Calculate Histogram of the data and Draw it
 * 
 */



void DrawHistogram(Mat data)
{
  Mat Histogram,NormalizedHistogram;
  
  Histogram = FindHistogram(data);
  
  double max_elem;
  int max_posi;
 
  FindMaxElementPosi(data,&max_elem,&max_posi);
  
  int histSize = (int)max_elem;
  //int histSize = 256;
  
  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
  
  //Histogram.convertTo(Histogram,CV_8UC1);

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(Histogram, Histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
 
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(Histogram.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(Histogram.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
  }
 
 
  /// Display
  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );
  
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/

/**
 * @function FindImageInverse
 * @param input Mat data(image)
 * @brief Calculate inverse of a given image (255 - image.data[i])
 * @return inverse image(Mat)
 */

Mat FindImageInverse(Mat image)
{
 
  Mat InverseImage = Mat(image.rows,image.cols,image.type());
 
  for(int i=0;i<image.rows;i++)
  {
    for(int j=0;j<image.cols;j++)
    {
      for(int k =0;k<image.channels();k++)
      {
	InverseImage.data[(i*image.cols+j)*image.channels()+k] = 255 - image.data[(i*image.cols+j)*image.channels()+k];
      }
    }
  }
  
  return(InverseImage);
}


/*-------------------------------------------------------------------------------------------------------------------------------------------*/

PCA train_pca, foreground_pca;
int nEigens;

void doPCA (const vector<Mat>& image, Mat Binary)
{
    nEigens = image.size();
    printf("image size %d\n",image.size());
    int sp,i,j,k,l;
    
    double temp_max_val;
  int temp_max_posi;
    
    for(int i=0;i<image.size();i++)
    {
       FindMaxElementPosi(image[0],&temp_max_val,&temp_max_posi);
	printf("Max value %lf\n",temp_max_val);
    }
    
   no_of_foregrnd_pix = NumberofForegroundPixel(Binary);
   printf("Number of foreground pixel %d\n",no_of_foregrnd_pix);
   Mat foreground_data (no_of_foregrnd_pix, image.size(), CV_32FC1);
  // binary_dst = binarization(src,2);
   k=0;
//    for (i = 0; i < image[0].cols*image[0].rows; i++)
//    {
//      if(binary_dst.data[i]!=255)
//      {
//        foreground_data.row(k).data[0]=image[0].data[i];
//        foreground_data.row(k).data[1]=image[1].data[i];
//        foreground_data.row(k).data[2]=image[2].data[i];
//        k++;
//      }
//    }
   for (i = 0; i < image[0].rows; i++)
   {
     for (j = 0; j < image[0].cols; j++)
     {
       if(Binary.at<uchar>(i,j)!=255)
       {
	 for(int l=0;l<image.size();l++)
	 {
	   foreground_data.at<float>(k,l) =(float) image[l].at<uchar>(i,j);
	   if(l == 0)
	   {
	     foreground_data.at<float>(k,l) = foreground_data.at<float>(k,l);
	   }
	 }
	 k = k + 1;
       }
     }
   }
   
  
    FindMaxElementPosi(foreground_data,&temp_max_val,&temp_max_posi);
    printf("Max value %lf\n",temp_max_val);
     FindMinElementPosi(foreground_data,&temp_max_val,&temp_max_posi);
    printf("Min value %lf\n",temp_max_val);
   
  // printf("no of foreground pixels are %d and %d\n",no_of_foregrnd_pix,k);
   
  // PCA foreground_pca;
   
   foreground_pca(foreground_data,Mat(),CV_PCA_DATA_AS_ROW,nEigens);
   
  // train_pca(train_data,Mat(),CV_PCA_DATA_AS_ROW,nEigens);
    
    //train_pca(hsv_planes,Mat(),CV_PCA_DATA_AS_COL,nEigens);
 
    float norma,hnorma,snorma,vnorma;
  
    int scalar_mul,h_src,s_src,v_src,temp1_scalar;
     
     
    float h_mean, s_mean, v_mean;
    float h_eigen,s_eigen,v_eigen,temp2_scalar;
    
   float total_var = 0.0;
   
   for(int i=0;i<nEigens;i++)
   {
     total_var = total_var + foreground_pca.eigenvalues.at<float>(i,0);
   }
  
 // h_eigen = train_pca.eigenvectors.row(0).data[0];
 // s_eigen = train_pca.eigenvectors.row(0).data[1];
 // v_eigen = train_pca.eigenvectors.row(0).data[2];
  
  
  for(int i=0;i<nEigens;i++)
   {
      float eval = foreground_pca.eigenvalues.at<float>(i,0);
      printf("eigenvalue is %f\n",eval);
      
      float per_var = eval/total_var;
      
      printf("percentage of variance is %f\n",per_var);
   }
   
   for(int i=0;i<nEigens;i++)
   {
     printf("%d eigenvectors  is\t",i);
     for(int j=0;j<image.size();j++)
     {
       printf("%f\t",foreground_pca.eigenvectors.at<float>(i,j));
     }
     printf("\n");
   }
    
  h_eigen = foreground_pca.eigenvectors.at<float>(0,0);
  s_eigen = foreground_pca.eigenvectors.at<float>(0,1);
  v_eigen = foreground_pca.eigenvectors.at<float>(0,2);
  
  temp2_scalar = (h_eigen * h_eigen) + (s_eigen * s_eigen) + (v_eigen * v_eigen);
  
  float denom;
  
  denom = sqrt(temp2_scalar);
  
 // printf("Max Eigen vector is \n%f\t%f\t%f\n",h_eigen,s_eigen,v_eigen);
  //fprintf(eigenvec,"\n%d\t%d\t%d\n",h_eigen,s_eigen,v_eigen);
  
  h_unitvec =  h_eigen / denom;
  s_unitvec =  s_eigen / denom;
  v_unitvec =  v_eigen / denom;
  
  
  printf("Unit eigen vector is \n%f\t%f\t%f\n",h_unitvec,s_unitvec,v_unitvec);
  //fprintf(unit_eigenvec,"\n%f\t%f\t%f\n",h_unitvec,s_unitvec,v_unitvec);

   
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/

Mat FindVoting(Mat LabelImage, int number)
{
  
  int total_pixel_count;
  Mat Voting = Mat::zeros(number,1,CV_32FC1);
  /*
  int *p;
  p = (int *)malloc(number*sizeof(int));
  for(int i =0;i<number;i++)
    p[i]=-1;
  int m = 0;
  int t;
  for(int i=0;i<LabelImage.rows;i++)
  {
    for(int j=0;j<LabelImage.cols;j++)
    {
      if(LabelImage.at<int>(i,j)!=(src.rows*src.cols))
      {
	if(m==number)
	  break;
	if(p[m]==-1)
	{
	  for(t=0;t<m;t++)
	  {
	    if(p[t]==LabelImage.at<int>(i,j))
	      break;
	  }
	  if(t==m)
	  {
	    p[m] = LabelImage.at<int>(i,j);
	    m = m + 1;
	  }
	}
      }
    }
    if(m==number)
      break;
  }
  for(int i =0;i<number;i++)
    printf("p value is %d\n",p[i]);
  */
 // printf("Voting initialization done\n");
  
  int k = 0;
  for(int i=0;i<LabelImage.rows;i++)
  {
    for(int j=0;j<LabelImage.cols;j++)
    {
      if(LabelImage.at<int>(i,j)!=number)
      {
	int temp = LabelImage.at<int>(i,j);
	Voting.at<float>(temp,0) = Voting.at<float>(temp,0) + 1;
	k = k + 1;
      }
    }
  }
  
  total_pixel_count = k;
 // printf("Voting initialization done and total pixel count is %d\n",total_pixel_count);
   
   
  for(int i=0;i<Voting.rows;i++)
  {
    for(int j=0;j<Voting.cols;j++)
      Voting.at<float>(i,j) = Voting.at<float>(i,j)/total_pixel_count;
  }
  
  
  return(Voting);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/



BeanStucture * CreateBean(int NumberOfBean, int MaxElement)
{
  BeanStucture *Beans;
  Beans = (BeanStucture *)malloc(NumberOfBean*sizeof(BeanStucture));
  
  int Middle;
  int k;
  k =(int) MaxElement/NumberOfBean;
  
  for(int i=0;i<NumberOfBean;i++)
  {
    Beans[i].BeanNumber = i;
    Beans[i].MaxElement = MaxElement;
    Beans[i].middle = i*k;
    Beans[i].UpperBound =(int) (Beans[i].middle + (k/2));
    Beans[i].UpperBound = Beans[i].UpperBound%MaxElement;
    Beans[i].LowerBound = (int) (Beans[i].middle - (k/2));
    Beans[i].LowerBound = MaxElement + Beans[i].LowerBound;
    Beans[i].LowerBound = Beans[i].LowerBound%MaxElement;
  }
  
  return(Beans);
}


int FindOpositeBean(int BeanNumber, int NumberOfBean)
{
  int OpositeBean;
  OpositeBean =(int) BeanNumber + NumberOfBean/2;
  OpositeBean = OpositeBean%NumberOfBean;
}

int FindBeanNumber(int angle, int MaxElement, int NumberOfBean)
{
  //int MaxElement = Beans[0].MaxElement;
  int BeanedAngle = MaxElement/NumberOfBean;
  int temp_angle =(int) angle + (BeanedAngle/2);
  int BeanNum;
  BeanNum =(int) (temp_angle/BeanedAngle);
  BeanNum = BeanNum % NumberOfBean;
  
  return(BeanNum);
}



Point2i FindNextPixel8Bean(Point2i p, int Bean)
{
  Point2i next;
  if(Bean == 0)
  {
    next.y = p.y;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 1)
  {
    next.y = p.y - 1;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 2)
  {
    next.y = p.y - 1;
    next.x = p.x;
    return(next);
  }
  else if(Bean == 3)
  {
    next.y = p.y - 1;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 4)
  {
    next.y = p.y;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 5)
  {
    next.y = p.y + 1;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 6)
  {
    next.y = p.y + 1;
    next.x = p.x;
    return(next);
  }
  else
  {
    next.y = p.y + 1;
    next.x = p.x + 1;
    return(next);
  }
}


Point2i FindNextPixel12Bean(Point2i p, int Bean)
{
  Point2i next;
  if(Bean == 0)
  {
    next.y = p.y;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 1)
  {
    next.y = p.y - 1;
    next.x = p.x + 2;
    return(next);
  }
  else if(Bean == 2)
  {
    next.y = p.y - 2;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 3)
  {
    next.y = p.y - 1;
    next.x = p.x;
    return(next);
  }
  else if(Bean == 4)
  {
    next.y = p.y - 2;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 5)
  {
    next.y = p.y - 1;
    next.x = p.x - 2;
    return(next);
  }
  else if(Bean == 6)
  {
    next.y = p.y;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 7)
  {
    next.y = p.y + 1;
    next.x = p.x - 2;
    return(next);
  }
  else if(Bean == 8)
  {
    next.y = p.y + 2;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 9)
  {
    next.y = p.y + 1;
    next.x = p.x;
    return(next);
  }
  else if(Bean == 10)
  {
    next.y = p.y + 2;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 11)
  {
    next.y = p.y + 1;
    next.x = p.x + 2;
    return(next);
  }
}


Point2i FindNextPixel16Bean(Point2i p, int Bean)
{
  Point2i next;
  if(Bean == 0)
  {
    next.y = p.y;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 1)
  {
    next.y = p.y - 1;
    next.x = p.x + 2;
    return(next);
  }
  else if(Bean == 2)
  {
    next.y = p.y - 1;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 3)
  {
    next.y = p.y - 2;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 4)
  {
    next.y = p.y - 1;
    next.x = p.x;
    return(next);
  }
  else if(Bean == 5)
  {
    next.y = p.y - 2;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 6)
  {
    next.y = p.y - 1;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 7)
  {
    next.y = p.y - 1;
    next.x = p.x - 2;
    return(next);
  }
  else if(Bean == 8)
  {
    next.y = p.y;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 9)
  {
    next.y = p.y + 1;
    next.x = p.x - 2;
    return(next);
  }
  else if(Bean == 10)
  {
    next.y = p.y + 1;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 11)
  {
    next.y = p.y + 2;
    next.x = p.x - 1;
    return(next);
  }
  else if(Bean == 12)
  {
    next.y = p.y + 1;
    next.x = p.x;
    return(next);
  }
  else if(Bean == 13)
  {
    next.y = p.y + 2;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 14)
  {
    next.y = p.y + 1;
    next.x = p.x + 1;
    return(next);
  }
  else if(Bean == 15)
  {
    next.y = p.y + 1;
    next.x = p.x + 2;
    return(next);
  }
}


Ray FindStrokeWidth(Point2i p,int Bean, Mat BoundaryImage, Mat GradBean, Mat GrayImage, int NumberOfBean)
{
  Ray TempRay;
  TempRay.p = p;
  TempRay.dist = 0;
  Point2i Next;
  Next = p;
  int OpositeBean = FindOpositeBean(Bean, NumberOfBean);
  int bean1,bean2;
  int m,n;
  while(1)
  {
    //Next = FindNextPixel8Bean(Next,Bean);
    
    if(NumberOfBean == 8)
      Next = FindNextPixel8Bean(Next,Bean);
    else if(NumberOfBean == 12)
      Next = FindNextPixel12Bean(Next,Bean);
    else if(NumberOfBean == 16)
      Next = FindNextPixel16Bean(Next,Bean);
	 
    
    if(validate(Next.y,Next.x,BoundaryImage.rows,BoundaryImage.cols))
    {
      if(GrayImage.at<uchar>(Next.y,Next.x) == 255)
      {
	for( m=Next.y-1;m<=Next.y+1;m++)
	{
	  for( n=Next.x-1;n<=Next.x+1;n++)
	  {
	    if(validate(m,n,BoundaryImage.rows,BoundaryImage.cols))
	    {
	      if(BoundaryImage.at<uchar>(m,n) == 0)
	      {
		Next.y = m;
		Next.x = n;
		break;
	      }
	    }
	  }
	  if(n<Next.x+2)
	   break;
	}
	if(m==Next.y+2 && n==Next.x+2)
	{
	  TempRay.dist = 0;
	  break;
	}
      }
      
      TempRay.points.push_back(Next);
      TempRay.dist = TempRay.dist + 1;
      if(BoundaryImage.at<uchar>(Next.y,Next.x) == 0)
      {
	bean1 = (OpositeBean + 1)%NumberOfBean;
	bean2 = (OpositeBean + NumberOfBean-1)%NumberOfBean;
	if(GradBean.at<int8_t>(Next.y,Next.x) == OpositeBean || GradBean.at<int8_t>(Next.y,Next.x) == bean1 || GradBean.at<int8_t>(Next.y,Next.x) == bean2)
	{
	  TempRay.q = Next;
	  TempRay.CalcEcluiDist();
	  break;
	}
	else
	{
	  TempRay.dist = 0;
	  break;
	}
      }
      
      if(TempRay.dist > (GrayImage.cols/3) || TempRay.dist > (GrayImage.rows/3))
      {
	TempRay.dist = 0;
	break;
      }
      
    }
    else
    {
      TempRay.dist = 0;
      break;
    }
  }
  
  
  return(TempRay);
}


vector<Ray> SWT(Mat Image, Mat BinaryImage)
{
  int row,col;
  row = Image.rows;
  col = Image.cols;
  
   int i,j; 
   
  Mat ForeGroundImage;
  ForeGroundImage = foreground_masked_image(Image, BinaryImage);
  
 
 
  Mat GrayImage;
  cvtColor(ForeGroundImage,GrayImage,CV_BGR2GRAY);
 // imshow("Gray",GrayImage);
  
 
  Mat BoundaryImage;
  BoundaryImage = boundaryextraction(BinaryImage);
  //imshow("boundary",BoundaryImage);
 
  
  int scale = 1;
  int delta = 0;
  int ddepth = CV_64F;
  
   /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( GrayImage, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  //convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( GrayImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
 // convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
 // addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  int no_edge_pixel = 0;
  for(int i=0;i<GrayImage.rows;i++)
  {
    for(int j=0;j<GrayImage.cols;j++)
    {
      if(BoundaryImage.data[i*col+j]==0)
      {
	no_edge_pixel = no_edge_pixel + 1;
      }
    }
  }
  
  Mat grad = Mat::zeros(GrayImage.rows,GrayImage.cols,CV_64FC1);
  Mat TempGradxyGradDirMag;
  TempGradxyGradDirMag = Mat(GrayImage.rows,GrayImage.cols,CV_64FC4);
  Mat NormalizeGradxy;
  NormalizeGradxy = Mat(GrayImage.rows,GrayImage.cols,CV_64FC2);
  int k = 0;
  double x_dir,y_dir;
  for(int i=0;i<GrayImage.rows;i++)
  {
    for(int j=0;j<GrayImage.cols;j++)
    {
      if(BoundaryImage.at<uchar>(i,j)==0)
      {
	x_dir = grad_x.at<double>(i,j);
	//TempGradx.at<double>(i,j) = x_dir;
	y_dir = grad_y.at<double>(i,j);
	//TempGrady.at<double>(i,j) = y_dir;
	grad.at<double>(i,j) = (atan2(y_dir, x_dir)*180)/PI;
	if(grad.at<double>(i,j) < 0)
	  grad.at<double>(i,j) = 180 - grad.at<double>(i,j);
	TempGradxyGradDirMag.at<Vec4d>(i,j)[0] = x_dir;
	TempGradxyGradDirMag.at<Vec4d>(i,j)[1] = y_dir;
	TempGradxyGradDirMag.at<Vec4d>(i,j)[2] = (atan2(y_dir, x_dir)*180)/PI;
	TempGradxyGradDirMag.at<Vec4d>(i,j)[3] = sqrt((x_dir*x_dir)+(y_dir*y_dir));
	NormalizeGradxy.at<Vec2d>(i,j)[0] = x_dir/TempGradxyGradDirMag.at<Vec4d>(i,j)[3];
	NormalizeGradxy.at<Vec2d>(i,j)[1] = y_dir/TempGradxyGradDirMag.at<Vec4d>(i,j)[3];
	
	//printf("grad x = %lf\t grad y = %lf and \tgrad dir = %lf\n",x_dir,y_dir,grad.at<double>(i,j));
      }
    }
  }
  
  double max_elem,min_elem;
  int posi;
  
  FindMaxElementPosi(grad,&max_elem,&posi);
  FindMinElementPosi(grad,&min_elem,&posi);
  
  //printf("MAX Angle is %lf\tand MIN Angle is %lf\n",max_elem,min_elem);
 
  
   
  Mat absgrad = Mat(grad.rows,grad.cols,CV_16UC1);
  
  
  for(int i=0;i<GrayImage.rows;i++)
  {
    for(int j=0;j<GrayImage.cols;j++)
    {
      if(BoundaryImage.at<uchar>(i,j)==0)
      {
	absgrad.at<u_int16_t>(i,j) =(u_int16_t) floor(grad.at<double>(i,j));
      }
    }
  }


//    namedWindow( "grad", CV_WINDOW_KEEPRATIO );
//    imshow("grad", grad);
//    waitKey(0);
  
  Mat abs_grad;
  convertScaleAbs( grad, abs_grad );
  //imshow("grad",abs_grad);

int flag = 1; 

  int BeanNum;
  int NumberOfBean;
  
  
  
  if(flag == 1)
    NumberOfBean = 8;
  else if(flag == 2)
    NumberOfBean = 12;
  else if(flag == 3)
    NumberOfBean = 16;
  else
  {
    printf("Bean not Selected Properly\n Choosing Default Bean = 8\n");
    NumberOfBean = 8;
  }
  
  
  BeanStucture *BeanData;
 
 BeanData = CreateBean(NumberOfBean,360);
  
  
  Mat GradBean = Mat(grad.rows,grad.cols,CV_8UC1);
 // Mat GradBeanHist = Mat::zeros(NumberOfBean,1,CV_32FC1);
  for(int i=0;i<GrayImage.rows;i++)
  {
    for(int j=0;j<GrayImage.cols;j++)
    {
      if(BoundaryImage.at<uchar>(i,j)==0)
      {
	  BeanNum = FindBeanNumber(absgrad.at<u_int16_t>(i,j),360,NumberOfBean);
	  GradBean.at<int8_t>(i,j) = BeanNum;
	//  GradBeanHist.at<float>(BeanNum,0) = GradBeanHist.at<float>(BeanNum,0) + 1;
      }
    }
  }
  
 
  int NumberOfStrokes = 0;
  vector<Ray> strokes1;
  for(int i=0;i<GrayImage.rows;i++)
  {
    for(int j=0;j<GrayImage.cols;j++)
    {
      if(BoundaryImage.at<uchar>(i,j)==0)
      {
	 int bean;
	 bean = GradBean.at<int8_t>(i,j);
	 Point2i p;
	 p.x = j;
	 p.y = i;
	 
	 Ray temp;
	 temp = FindStrokeWidth(p,bean,BoundaryImage,GradBean,GrayImage, NumberOfBean);
	 if(temp.dist > 2)
	 {
	  strokes1.push_back(temp);
	  NumberOfStrokes = NumberOfStrokes + 1;
	 }
      }
    }
  }
 
 //printf("NumberOfStrokes is %d\n",NumberOfStrokes);

  return (strokes1);
}


/*-------------------------------------------------------------------------------------------------------------------------------------------*/

double CalculateZValue(double SampleMean, double PopulationMean, double PopulationSD, int SampleSize)
{
  double z;
  z = SampleMean - PopulationMean;
  double temp = sqrt(SampleSize);
  temp = PopulationSD/temp;
  z = z/temp;
  return(z);
}

double CalculateZValue(double SampleMean, double PopulationMean, double SEM)
{
  double z;
  z = SampleMean - PopulationMean;
  z = z / SEM;
  return(z);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/

double oneSampleTtest(double sample_mean, double population_mean, double sample_sd, double sample_size)
{
  double t;
  double sigma;
  sigma = sample_sd/(sqrt(sample_size));
  t = sample_mean - population_mean;
  t = t / sigma;
  return(t);
}

double twoSampleTtest(int sample1_size, int sample2_size, double sample1_mean, double sample2_mean, double sample1_sd, double sample2_sd)
{
  double sigma;
  double sigma1,sigma2;
  sigma1 = sample1_sd * sample1_sd;
  sigma1 = sigma1/sample1_size;
  sigma2 = sample2_sd * sample2_sd;
  sigma2 = sigma2/sample2_size;
  sigma = sigma1 + sigma2;
  sigma = sqrt(sigma);
  double t;
  t = sample1_mean - sample2_mean;
  t = t/sigma;
  return(t);
}


/*-------------------------------------------------------------------------------------------------------------------------------------------*/


/*---------------------------------------------------------------AdjacencyMatrix----------------------------------------------------------------------------*/




/*@function FindAdjacencyMatrix
 * @param : input : Mat RelationMat - A adjacency matrix with relation value between two node is given, in double
 * 		    double connection_condition - It is the value by checking which two nodes are connected
 * 
 * example : connection_condition = x;
 * 	     if(RelationMat.at<double>(i,j) > x)
 * 		
 * return : A AdjacencyMatrix of bool type (vector<vector<bool> >)
 * */

vector<vector<bool> > FindAdjacencyMatrix(Mat RelationMat, double connection_condition)
{
  vector<vector<bool> > AdjMat;
  for(int i=0;i<RelationMat.rows;i++)
  {
    vector<bool> t;
    for(int j=0;j<RelationMat.cols;j++)
    {
      
      if(RelationMat.at<double>(i,j) > connection_condition)
      {
	t.push_back(true);
      }
      else
      {
	t.push_back(false);
      }
    }
    AdjMat.push_back(t);
    t.clear();
  }
  return(AdjMat);
}



vector<vector<bool> > FindAdjacencyMatrix2data(Mat RelationMat1, double connection_condition1, Mat RelationMat2, double connection_condition2)
{
  
  vector<vector<bool> > AdjMat;
  for(int i=0;i<RelationMat1.rows;i++)
  {
    vector<bool> t;
    for(int j=0;j<RelationMat1.cols;j++)
    {
      
      if(RelationMat1.at<double>(i,j) > connection_condition1 && RelationMat2.at<double>(i,j) > connection_condition2)
      {
	t.push_back(true);
      }
      else
      {
	t.push_back(false);
      }
    }
    AdjMat.push_back(t);
    t.clear();
  }
  return(AdjMat);
  
  /*
  vector<vector<bool> > AdjMat;
  vector<vector<bool> > AdjMat1;
  vector<vector<bool> > AdjMat2;
  
  AdjMat1  = FindAdjacencyMatrix(RelationMat1, connection_condition1);
  AdjMat2 = FindAdjacencyMatrix(RelationMat2, connection_condition2);
  
  for(int i=0;i<RelationMat1.rows;i++)
  {
    vector<bool> t1 = AdjMat1[i];
    vector<bool> t2 = AdjMat2[i];
    vector<bool> t;
    for(int j=0;j<RelationMat1.cols;j++)
    {
      if(t1[j] == true && t2[j] == true)
	t.push_back(true);
      else
	t.push_back(false);
    }
    AdjMat.push_back(t);
    t1.clear();
    t2.clear();
    t.clear();
  }
  AdjMat1.clear();
  AdjMat2.clear();
  return(AdjMat);
  */
 
}




/*-------------------------------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------------------------------*/





void GiveLabelDFS(vector<vector<bool> > AdjMat, vector<int> &tempcomponent, vector<int> &labels, vector<bool> &ccflag)
{
  vector<bool> tempadj;
  tempadj = AdjMat[tempcomponent.back()];
  for(int i=0;i<tempadj.size();i++)
  {
    if(ccflag[i] == false && tempadj[i] == true)
    {
      ccflag[i] = true;
      labels[i] = labels[tempcomponent.back()];
      tempcomponent.push_back(i);
      GiveLabelDFS(AdjMat,tempcomponent,labels,ccflag);
    }
  }
}








/*@function : 		DFSCC 
 * @desc :		Find ConnectedComponent based on depth First search of the adjacency matrix
 * @param:	input:	vector<vector<bool> > AdjMat :	AdjacencyMatrix in bool format
 * 
 * 			vector<vector<int> > &component: address to the component
 * 
 * */

void DFSCC(vector<vector<bool> > AdjMat, vector<vector<int> > &component, vector<int> &labels)
{
  vector<bool> ccflag;
  for(int i=0;i<AdjMat.size();i++)
  {
    ccflag.push_back(false);
  }
  
  int label = 0;
  
  
  for(int i=0;i<AdjMat.size();i++)
  {
    if(ccflag[i] == false)
    {
      vector<int> tempconnected;
     // printf("In DFSCC i = %d\n",i);
      tempconnected.push_back(i);
      labels[i] = label;
      ccflag[i] = true;
      GiveLabelDFS(AdjMat, tempconnected, labels, ccflag);      
      component.push_back(tempconnected);
      label = label + 1;
    }
  }
  
  
}


/*-------------------------------------------------------------------------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------------------------------------------------------------------------*/



#if _SVM_

vector<ConfusionMatrix> segmentation_withSVM(vector<FeatureVec> FV, vector<int> GTLabels, vector<int> ClusterLabels, TDC Data, int algo)
{ 
  int C = 6;
  
    //(1)-(2)separable and not sets
            CvSVMParams params;
            params.svm_type = CvSVM::C_SVC;
            params.kernel_type = CvSVM::POLY; //CvSVM::LINEAR;
            params.degree = 0.5;
            params.gamma = 1;
            params.coef0 = 1;
            params.C = C;
            params.nu = 0.5;
            params.p = 0;
            params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
  
   
  
  
    vector<ConfusionMatrix> CM_SVM(Data.ClassNumber.size());
    
    for(int i=0;i<CM_SVM.size();i++)
    {
      CM_SVM[i].initialize();
      CM_SVM[i].initializeMulticlass(Data.ClassNumber.size());
    }
    
       // learn classifier
#if defined HAVE_OPENCV_OCL && _OCL_SVM_
    cv::ocl::CvSVM_OCL svmClassifier(trainSamples, trainClasses, Mat(), Mat(), params);
#else
    CvSVM svmClassifier( Data.TrainData, Data.TrainClass, Mat(), Mat(), params );
#endif
    
    
    printf("In SVM \n");
    
    vector<int> PredictedLabels;
    
    
    for(int i=0;i<FV.size();i++)
    {
      
	FeatureVec elem = FV[i];
	Mat TestData;
	Mat(elem).copyTo(TestData);
	//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
	transpose(TestData,TestData);
	//TestData = TestData.reshape( 1, TestData.rows );
	//TestData.convertTo( TestData, CV_32FC1 );
	
	//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
	
	int response = (int)svmClassifier.predict( TestData );
	
	PredictedLabels.push_back(response);

    }
    
  
    int ncluster = 0;
    
    for(int i=0;i<ClusterLabels.size();i++)
    {
      if(ncluster<ClusterLabels[i])
	ncluster = ClusterLabels[i];
    }
    
    vector<vector<int> > HistClusterClass(ncluster+1);
    
    for(int i=0;i<HistClusterClass.size();i++)
    {
      vector<int> temp(Data.ClassNumber.size(),0);
      HistClusterClass[i] = temp;
    }
    
    for(int i=0;i<PredictedLabels.size();i++)
    {
      HistClusterClass[ClusterLabels[i]][PredictedLabels[i]] =  HistClusterClass[ClusterLabels[i]][PredictedLabels[i]] + 1;
    }
    
    vector<int> ClusterClass(HistClusterClass.size());
    
    for(int i=0;i<HistClusterClass.size();i++)
    {
      vector<int> temp = HistClusterClass[i];
      int max = 0;
      for(int j=0;j<temp.size();j++)
      {
	if(temp[max] < temp[j])
	  max = j;
      }
      ClusterClass[i] = max;
    }
    
   
   for(int i=0;i<FV.size();i++)
   {
     int response = ClusterClass[ClusterLabels[i]];
     int GtClass = GTLabels[i];
     
     int gtposi,resposi;
	
	for(int p=0;p<Data.ClassNumber.size();p++)
	{
	  if(GtClass == Data.ClassNumber[p])
	    gtposi = p;
	  if(response == Data.ClassNumber[p])
	    resposi = p;
	}
	
	CM_SVM[resposi].multiclassCM[gtposi] = CM_SVM[resposi].multiclassCM[gtposi] + 1;
	//printf("Response = %d\t gtlabel = %d\n",response,GtClass);
	

     
     if(algo == 1)
       MultiClassResSCCN[resposi][gtposi] = MultiClassResSCCN[resposi][gtposi] + 1;
     if(algo == 2)
       MultiClassResSCE[resposi][gtposi] = MultiClassResSCE[resposi][gtposi] + 1;
   
      for(int j=0;j<Data.ClassNumber.size();j++)
	  {
	    ConfusionMatrix cm_svm_j = CM_SVM[j];
	    
	    
	    if(response == Data.ClassNumber[j]) // response == A Class j (Positive part (tp,fp))
	    {
	      if(Data.ClassNumber[j] == GTLabels[i]) // GT == class j => tp
	      {
		cm_svm_j.tp = cm_svm_j.tp + 1;
		CM_SVM[j] = cm_svm_j;
	      }
	      else // GT != class j => fp
	      {
		cm_svm_j.fp = cm_svm_j.fp + 1;
		CM_SVM[j] = cm_svm_j;
	      }
	    }
	    else // response != Class j (Negetive part (tn,fn))
	    { 
	      if(Data.ClassNumber[j] == PredictedLabels[i]) // GT == class j ==> fn
	      {
		cm_svm_j.fn = cm_svm_j.fn + 1;
		CM_SVM[j] = cm_svm_j;
	      }
	      else // GT != class =>  tn
	      {
		cm_svm_j.tn = cm_svm_j.tn + 1;
		CM_SVM[j] = cm_svm_j;
	      }
	    }
	    
	  }
    }
    

    
    return(CM_SVM);
    
}

#endif


  


/*---------------------------------------------------------------------GUI Adjacency----------------------------------------------------------------------*/

Mat adjdata;
int datathreshold = 5;
int data2threshold = 5;
Mat adjdata2;
int const maxthreshold = 100;
vector<Rect> boundaryRect;
vector<int> datarectmap;
vector<vector<Point> > bounding_poly;
Mat clusterimage;

void AdjacencyThreshold( int, void* )
{
  vector<vector<bool> > AdjMat;
  
  double th =(double) datathreshold/100;
  AdjMat = FindAdjacencyMatrix(adjdata, th);
  vector<vector<int> > adjcc;
  vector<int> adjlabels(adjdata.rows);
  DFSCC(AdjMat, adjcc, adjlabels);
  
  vector<vector<int> > ccColor;
  
  for(int i=0;i<adjcc.size();i++)
  {
    //Scalar cl = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    vector<int> cl;
    for(int j=0;j<3;j++)
      cl.push_back(rng.uniform(0, 255));
    ccColor.push_back(cl);
    cl.clear();
  }
  
  int p,q;
	  for(int i=0;i<adjlabels.size();i++)
	  {
	      int j = datarectmap[i];
	      p = 0;
	      for(int m=boundaryRect[j].y;m<boundaryRect[j].y+boundaryRect[j].height;m++)
	      {
		q=0;
		for(int n=boundaryRect[j].x;n<boundaryRect[j].x+boundaryRect[j].width;n++)
		{
		  int temp_col = boundaryRect[j].width;
		  bool measure_dist;
		  if((pointPolygonTest(bounding_poly[j],Point(n,m),measure_dist) > 0.0) && binary_dst.data[m*binary_dst.cols+n]==0)
		  {
		   // image.at<Vec3b>(m,n)[0] = uniform_background.at<Vec3b>(m,n)[0];
		   // image.at<Vec3b>(m,n)[1] = uniform_background.at<Vec3b>(m,n)[1];
		   // image.at<Vec3b>(m,n)[2] = uniform_background.at<Vec3b>(m,n)[2];
		   // image.at<Vec3b>(m,n) = ccColor[adjlabels[i]];
		    vector<int> tmpcl;
		    tmpcl = ccColor[adjlabels[i]];
		    for(int k=0;k<3;k++)
		    {
		      clusterimage.at<Vec3b>(m,n)[k] = tmpcl[k];
		    }
		    tmpcl.clear();
		  }
		  q++;
		}
		p++;
	      }
	    }
	    imshow("Cluster",clusterimage);
  
}

Mat GUIAdjCC(Mat Data, vector<Rect> boundRect, vector<vector<Point> > contours_poly, vector<int> DataRectMap)
{
  Data.copyTo(adjdata);
  Data.release();
  boundaryRect = boundRect;
  boundRect.clear();
  datarectmap = DataRectMap;
  DataRectMap.clear();
  bounding_poly = contours_poly;
  contours_poly.clear();
  
  clusterimage = Mat(src.rows,src.cols,src.type(),Scalar(255,255,255));
  
  
  namedWindow( "Cluster", CV_WINDOW_KEEPRATIO );
  createTrackbar( "Threshold",
			    "Cluster", & datathreshold,
			    maxthreshold, AdjacencyThreshold );
  
  
   /// Call the function to initialize
	    AdjacencyThreshold( 0, 0 );
	    waitKey(0);
	    destroyWindow("Cluster");
	    
	    
	    return(clusterimage);
  
}



void Adjacency2Threshold( int, void* )
{
  vector<vector<bool> > AdjMat;
  
  double th1 =(double) datathreshold/100;
  double th2 =(double) data2threshold/100; 
  AdjMat = FindAdjacencyMatrix2data(adjdata, th1, adjdata2, th2);
  vector<vector<int> > adjcc;
  vector<int> adjlabels(adjdata.rows);
  DFSCC(AdjMat, adjcc, adjlabels);
  
  vector<vector<int> > ccColor;
  
  for(int i=0;i<adjcc.size();i++)
  {
    //Scalar cl = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    vector<int> cl;
    for(int j=0;j<3;j++)
      cl.push_back(rng.uniform(0, 255));
    ccColor.push_back(cl);
    cl.clear();
  }
  
  int p,q;
	  for(int i=0;i<adjlabels.size();i++)
	  {
	      int j = datarectmap[i];
	      p = 0;
	      for(int m=boundaryRect[j].y;m<boundaryRect[j].y+boundaryRect[j].height;m++)
	      {
		q=0;
		for(int n=boundaryRect[j].x;n<boundaryRect[j].x+boundaryRect[j].width;n++)
		{
		  int temp_col = boundaryRect[j].width;
		  bool measure_dist;
		  if((pointPolygonTest(bounding_poly[j],Point(n,m),measure_dist) > 0.0) && binary_dst.data[m*binary_dst.cols+n]==0)
		  {
		   // image.at<Vec3b>(m,n)[0] = uniform_background.at<Vec3b>(m,n)[0];
		   // image.at<Vec3b>(m,n)[1] = uniform_background.at<Vec3b>(m,n)[1];
		   // image.at<Vec3b>(m,n)[2] = uniform_background.at<Vec3b>(m,n)[2];
		   // image.at<Vec3b>(m,n) = ccColor[adjlabels[i]];
		    vector<int> tmpcl;
		    tmpcl = ccColor[adjlabels[i]];
		    for(int k=0;k<3;k++)
		    {
		      clusterimage.at<Vec3b>(m,n)[k] = tmpcl[k];
		    }
		    tmpcl.clear();
		  }
		  q++;
		}
		p++;
	      }
	    }
	    imshow("Cluster",clusterimage);
  
}


Mat GUIAdjCC2data(Mat Data1, Mat Data2, vector<Rect> boundRect, vector<vector<Point> > contours_poly, vector<int> DataRectMap)
{
  Data1.copyTo(adjdata);
  Data1.release();
  Data2.copyTo(adjdata2);
  Data2.release();
  boundaryRect = boundRect;
  boundRect.clear();
  datarectmap = DataRectMap;
  DataRectMap.clear();
  bounding_poly = contours_poly;
  contours_poly.clear();
  
  clusterimage = Mat(src.rows,src.cols,src.type(),Scalar(255,255,255));
  
  
  namedWindow( "Cluster", CV_WINDOW_KEEPRATIO );
  createTrackbar( "Proj",
			    "Cluster", & datathreshold,
			    maxthreshold, Adjacency2Threshold );
  
  createTrackbar( "SW",
			    "Cluster", & data2threshold,
			    maxthreshold, Adjacency2Threshold );
  
  
   /// Call the function to initialize
	    Adjacency2Threshold( 0, 0 );
	    waitKey(0);
	    destroyWindow("Cluster");
	    
	    
	    return(clusterimage);
  
}


/*-------------------------------------------------------------------------------------------------------------------------------------------*/

vector<int> FindIntersection(vector<int> &v1, vector<int> &v2)
{
  vector<int> v3;
  
  sort(v1.begin(), v1.end());
  sort(v2.begin(), v2.end());
  
  set_intersection(v1.begin(),v1.end(),v2.begin(),v2.end(),back_inserter(v3));
  
  return (v3);
}



/*-------------------------------------------------------------------------------------------------------------------------------------------*/


void CalculateAccuracyAll(vector<cluster> &GT, vector<cluster> &seg, int *tp, int *tn, int *fp, int *fn, float *ri, float *ari, float *pre, float *rec, float *fs)
{
 // printf("In Accuracy Calculation\n");
	vector<vector<int> > ClassFreqPerCluster;
	
	int size = 0;
	for(int i=0;i<seg.size();i++)
	{
	  size = size + seg[i].blocks.size();
	}
	
	//printf("size = %d\n",size);
  
  vector<int> CB(size);
  vector<int> KB(size);
  
  for(int i=0;i<seg.size();i++)
  {
    for(int j=0;j<seg[i].blocks.size();j++)
    {
      KB[seg[i].blocks[j]] = i;
    }
  }
  
  for(int i=0;i<GT.size();i++)
  {
    for(int j=0;j<GT[i].blocks.size();j++)
    {
      CB[GT[i].blocks[j]] = i;
    }
  }
  int total;
  if(size > 1)
  {
    int to;
    if(calcCNR(size,2,&to))
	{
	 // printf("NCR calculated successfully\n");
	  printf("%d Choose 2 = %d\n",size,to);
	  total = to;
	}
	else
	{
	  total = 0;
	//  printf("NCR calculation failed\n");
	}
  }
  else
    total = 1;
  
  
    
    
    
    int tp_fp = 0;
    int tp_fn = 0;
    
    for(int i=0;i<GT.size();i++)
    {
		int to;
		if(GT[i].blocks.size() > 1)
		{
			if(calcCNR(GT[i].blocks.size(),2,&to))
			{
			 // printf("NCR calculated successfully\n");
			  printf("%d Choose 2 = %d\n",GT[i].blocks.size(),to);
			  tp_fn = tp_fn + to;
			}
		}
		else
			tp_fn = tp_fn + 0;
	}
	
	
	for(int i=0;i<seg.size();i++)
    {
		vector<int> freq(GT.size());
		for(int j=0;j<GT.size();j++)
			freq[j] = 0;
		for(int j=0;j<seg[i].blocks.size();j++)
		{
			freq[CB[seg[i].blocks[j]]] = freq[CB[seg[i].blocks[j]]] + 1;
		}
		ClassFreqPerCluster.push_back(freq);
		
		int to;
		if(seg[i].blocks.size() > 1)
		{
			if(calcCNR(seg[i].blocks.size(),2,&to))
			{
			//  printf("NCR calculated successfully\n");
			  printf("%d Choose 2 = %d\n",seg[i].blocks.size(),to);
			  tp_fp = tp_fp + to;
			}
		}
		else
			tp_fp = tp_fp + 0;
	}
	
	
	
	int truep,truen,falsep,falsen;
	
	truep = 0;
	
	for(int i=0;i<seg.size();i++)
	{
		for(int j=0;j<GT.size();j++)
		{
			if(ClassFreqPerCluster[i][j] > 1)
			{
				int to;
				if(calcCNR(ClassFreqPerCluster[i][j],2,&to))
				{
					//printf("NCR calculated successfully\n");
					printf("%d Choose 2 = %d\n",ClassFreqPerCluster[i][j],to);
					truep = truep + to;
				}
				else
				  truep = truep + 0;
			}
		}
	}
	
	ClassFreqPerCluster.clear();
	CB.clear();
	KB.clear();
	
	falsep = tp_fp - truep;
	falsen = tp_fn - truep;
	truen = total - (truep + falsep + falsen);
	
	*tp = truep;
	*fp = falsep;
	*fn = falsen;
	*tn = truen;
	
	
  float RandIndex = (truep + truen) * 1.0;
  RandIndex = RandIndex/(truep + truen + falsep + falsen);
  *ri = RandIndex;
  
  /*
   * 
   * //calculate ari by formulae
  float ARIn;
  float ARI_deno = 2 * ((truep * truen) - (falsep * falsen)) * 1.0;
  float ARI_neu = ((truen + falsen) * (falsen + truep)) + ((truen + falsep) * (falsep + truep));
  if(ARI_deno == ARI_neu)
    ARIn = 1;
  else
    ARIn = ARI_deno/ARI_neu;
  */
  
  int indexi = 0;
  
  for(int i=0;i<GT.size();i++)
  {
    for(int j=0;j<seg.size();j++)
    {
      vector<int> inter;
      inter = FindIntersection(GT[i].blocks,seg[j].blocks);
      printf("inter size %d\n",inter.size());
      if(inter.size() > 1)
      {
	int to;
	if(calcCNR(inter.size(),2,&to))
	{
	  printf("%d Choose 2 = %d\n",inter.size(),to);
	  indexi = indexi + to;
	}
	else
	  indexi = indexi + 0;
      }
    }
  }
  
  
  
  
  float IndexVal = indexi * 1.0;
  float ExpectedRI = ((tp_fp) * (tp_fn) * 1.0)/total;
  float MaxRI = ((tp_fp + tp_fn) * 1.0)/2;
  float ARIn;
  float ARI_deno = IndexVal - ExpectedRI;
  float ARI_neu = MaxRI - ExpectedRI;
  if(ARI_deno == ARI_neu)
    ARIn = 1.0;
  else
    ARIn = ARI_deno/ARI_neu;
  
  printf("ARI =  (%f - %f)/(%f - %f) = %f",IndexVal,ExpectedRI,MaxRI,ExpectedRI,ARIn);
  
  *ari = ARIn;
  
  float precesion =(float) (truep*1.0)/(truep + falsep);
  *pre = precesion;
  float Recall = truep * 1.0;
  Recall = Recall/(truep + falsen);
  *rec = Recall;
  
  int B = 1;
  float FScore = (((B * B) + 1) * precesion * Recall)/(((B * B) * precesion) + Recall);
  *fs = FScore;
  printf("Total Number = %d\n",total);
  printf("Tp+FP = %d\n",tp_fp);
  printf("Tp+FN = %d\n",tp_fn);
  printf("tp = %d\t fp = %d\t tn = %d\t fn = %d\n",truep,falsep,truen,falsen);
  printf("RI %f\t ARI %f\n",RandIndex,ARIn);
  printf("precesion %f\tRecall %f\tFScore %f \n",precesion,Recall,FScore);
	
}



void CalculateAccuracy(vector<cluster> &GT, vector<cluster> &seg, vector<vector<int> > &classpercluster, int *tp, int *tn, int *fp, int *fn)
{
  
  
  // printf("In Accuracy Calculation\n");
	vector<vector<int> > ClassFreqPerCluster;
	
	int size = 0;
	for(int i=0;i<seg.size();i++)
	{
	  size = size + seg[i].blocks.size();
	}
	
	//printf("size = %d\n",size);
  
  vector<int> CB(size);
  vector<int> KB(size);
  
  for(int i=0;i<seg.size();i++)
  {
    for(int j=0;j<seg[i].blocks.size();j++)
    {
      KB[seg[i].blocks[j]] = i;
    }
  }
  
  for(int i=0;i<GT.size();i++)
  {
    for(int j=0;j<GT[i].blocks.size();j++)
    {
      CB[GT[i].blocks[j]] = i;
    }
  }
  int total;
  if(size > 1)
  {
    int to;
    if(calcCNR(size,2,&to))
	{
	 // printf("NCR calculated successfully\n");
	  printf("%d Choose 2 = %d\n",size,to);
	  total = to;
	}
	else
	{
	  total = 0;
	//  printf("NCR calculation failed\n");
	}
  }
  else
    total = 1;
  
  
    
    
    
    int tp_fp = 0;
    int tp_fn = 0;
    
    for(int i=0;i<GT.size();i++)
    {
		int to;
		if(GT[i].blocks.size() > 1)
		{
			if(calcCNR(GT[i].blocks.size(),2,&to))
			{
			 // printf("NCR calculated successfully\n");
			  printf("%d Choose 2 = %d\n",GT[i].blocks.size(),to);
			  tp_fn = tp_fn + to;
			}
		}
		else
			tp_fn = tp_fn + 0;
	}
	
	
	for(int i=0;i<seg.size();i++)
    {
		vector<int> freq(GT.size());
		for(int j=0;j<GT.size();j++)
			freq[j] = 0;
		for(int j=0;j<seg[i].blocks.size();j++)
		{
			freq[CB[seg[i].blocks[j]]] = freq[CB[seg[i].blocks[j]]] + 1;
		}
		ClassFreqPerCluster.push_back(freq);
		
		int to;
		if(seg[i].blocks.size() > 1)
		{
			if(calcCNR(seg[i].blocks.size(),2,&to))
			{
			//  printf("NCR calculated successfully\n");
			  printf("%d Choose 2 = %d\n",seg[i].blocks.size(),to);
			  tp_fp = tp_fp + to;
			}
		}
		else
			tp_fp = tp_fp + 0;
	}
	
	
	
	int truep,truen,falsep,falsen;
	
	truep = 0;
	
	for(int i=0;i<seg.size();i++)
	{
		for(int j=0;j<GT.size();j++)
		{
			if(ClassFreqPerCluster[i][j] > 1)
			{
				int to;
				if(calcCNR(ClassFreqPerCluster[i][j],2,&to))
				{
					//printf("NCR calculated successfully\n");
					printf("%d Choose 2 = %d\n",ClassFreqPerCluster[i][j],to);
					truep = truep + to;
				}
				else
				  truep = truep + 0;
			}
		}
	}
	
	ClassFreqPerCluster.clear();
	CB.clear();
	KB.clear();
	
	falsep = tp_fp - truep;
	falsen = tp_fn - truep;
	truen = total - (truep + falsep + falsen);
	
	*tp = truep;
	*fp = falsep;
	*fn = falsen;
	*tn = truen;
	
  
  
  printf("tp = %d\t fp = %d\t tn = %d\t fn = %d\n",truep,falsep,truen,falsen);
  
  
}



float CalculateRandIndex(vector<cluster> &GT, vector<cluster> &seg, vector<vector<int> > &classpercluster)
{
  int truep,truen,falsep,falsen;
  CalculateAccuracy(GT,seg,classpercluster,&truep,&truen,&falsep,&falsen);  
  float RandIndex = (truep + truen) * 1.0;
  RandIndex = RandIndex/(truep + truen + falsep + falsen);
  return(RandIndex);
  
}

float CalculateARI(vector<cluster> &GT, vector<cluster> &seg, vector<vector<int> > &classpercluster)
{
  int truep,truen,falsep,falsen;
  CalculateAccuracy(GT,seg,classpercluster,&truep,&truen,&falsep,&falsen);
  float ARI;
  float ARI_deno = 2 * ((truep * truen) - (falsep * falsen)) * 1.0;
  float ARI_neu = ((truen + falsen) * (falsen + truep)) + ((truen + falsep) * (falsep + truep));
  ARI = ARI_deno/ARI_neu;
  return(ARI);
}

float CalculatePrecesion(vector<cluster> &GT, vector<cluster> &seg, vector<vector<int> > &classpercluster)
{
  int truep,truen,falsep,falsen;
  CalculateAccuracy(GT,seg,classpercluster,&truep,&truen,&falsep,&falsen);
  float precesion =(float) (truep*1.0)/(truep + falsep);
  return(precesion);
  
}

float CalculateRecall(vector<cluster> &GT, vector<cluster> &seg, vector<vector<int> > &classpercluster)
{
  int truep,truen,falsep,falsen;
  CalculateAccuracy(GT,seg,classpercluster,&truep,&truen,&falsep,&falsen);
  float Recall = truep * 1.0;
  Recall = Recall/(truep + falsen);
  return(Recall);
  
}

float CalculateFscore(vector<cluster> &GT, vector<cluster> &seg, vector<vector<int> > &classpercluster, int B)
{
  float fscore,Precesion,Recall;
  Precesion = CalculatePrecesion(GT,seg,classpercluster);
  Recall = CalculateRecall(GT,seg,classpercluster);
  fscore = (((B * B) + 1) * Precesion * Recall)/(((B * B) * Precesion) + Recall);
  return(fscore);
}



/*-------------------------------------------------------------------------------------------------------------------------------------------*/


double CalculatePurity(vector<cluster> &GT, vector<cluster> &seg)
{
  int total_num = 0;
  double purity = 0.0;
  

  
  for(int i=0;i<seg.size();i++)
  {
    int max = 0;
    total_num = total_num + seg[i].blocks.size();
    for(int j=0;j<GT.size();j++)
    {
      vector<int> inter;
      inter = FindIntersection(GT[j].blocks,seg[i].blocks);
      if(!inter.empty())
      {
	if(max<inter.size())
	  max = inter.size();
      }
    }
    purity = purity + (double) max;
  }
 
  purity = purity/total_num;
  
  printf("Purity %lf\n",purity);
  return(purity);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/

double CalculateEntropy(vector<cluster> &C)
{
  double entropy = 0.0;
  int size = 0;
  for(int i=0;i<C.size();i++)
  {
    size =  size + C[i].blocks.size();
  }
 // printf("size = %d\n",size);
  for(int i=0;i<C.size();i++)
  {
    if(!C[i].blocks.empty())
    {
     // printf("block size = %d\n",C[i].blocks.size());
      double temp;
      temp = (C[i].blocks.size()*1.0)/size;
      double lval = log2(temp);
      temp = temp * lval;
      entropy = entropy + lval;
    }
  }
  if(entropy != 0)
    entropy = entropy * -1.0;
  printf("Entropy is %lf\n",entropy);
  
  return(entropy);
}

/*
 * @func: CalculateConditionalEntropy : calculate conditional entropy of C when K is given
 * */

double CalculateConditionalEntropy(vector<cluster> &C, vector<cluster> &K)
{
  double CE = 0.0;
  
  int size = 0;
  for(int i=0;i<C.size();i++)
  {
    size =  size + C[i].blocks.size();
  }
  
  
  for(int c=0;c<C.size();c++)
  {
    for(int k =0;k<K.size();k++)
    {
      if(!C[c].blocks.empty() && !K[k].blocks.empty())
      {
	vector<int> inter;
	inter = FindIntersection(C[c].blocks,K[k].blocks);
	if(!inter.empty())
	{
	  double tval = (inter.size() * 1.0)/size;
	  double deno = tval;
	  double neu = (K[k].blocks.size() * 1.0)/size;
	  double lval = deno/neu;
	  lval = log2(lval);
	  tval = tval * lval;
	  CE = CE + tval;
	}
      }
    }
  }
  
  if(CE != 0)
    CE = CE * -1.0;
  printf("Conditional Entropy is %lf\n",CE);
  
  return(CE);
  
}

double CalculateMutualInformation(vector<cluster> &GT, vector<cluster> &seg)
{
  double MI = 0.0;
  
  int size = 0;
  for(int i=0;i<GT.size();i++)
  {
    size =  size + GT[i].blocks.size();
  }
  
  for(int i=0;i<GT.size();i++)
  {
    for(int j=0;j<seg.size();j++)
    {
      if(!GT[i].blocks.empty() && !seg[j].blocks.empty())
      {
	vector<int> inter;
	inter = FindIntersection(GT[i].blocks,seg[j].blocks);
	if(!inter.empty())
	{
	 // printf("Inter size %d\n",inter.size());
	  double tval = (inter.size()*1.0)/size;
	  double deno = size * inter.size() * 1.0;
	  double neu = GT[i].blocks.size() * seg[j].blocks.size() * 1.0;
	  double lval = deno/neu;
	  lval = log2(lval);
	  tval = tval * lval;
	  MI = MI + tval;
	}
      }
    }
  }
  
  printf("Mutual Information %lf\n",MI);

  return(MI);
  
}

double CalculateNMI(vector<cluster> &GT, vector<cluster> &seg)
{
  double MI = CalculateMutualInformation(GT,seg);
  double NMI;
  double deno = MI;
  double neu = (CalculateEntropy(GT) + CalculateEntropy(seg))/2;
  if(deno == neu)
    NMI = 1;
  else
    NMI = deno/neu;
  if(MI == 0.0)
    NMI = 0.0;
  printf("Normalized Mutual Information %lf\n",NMI);
  return(NMI);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/

double CalculateHomogeneity(vector<cluster> &GT, vector<cluster> &seg)
{
  double H_GT = 0;
  int size = 0;
  for(int i=0;i<GT.size();i++)
  {
    size = size + GT[i].blocks.size();
  }
  
  
  double h;
  
  H_GT = CalculateEntropy(GT);
  double C_entropy_gt_seg = CalculateConditionalEntropy(GT,seg);
  if(C_entropy_gt_seg == 0)
    h = 1;
  else
    h =1 - (C_entropy_gt_seg/H_GT);
  
  printf("Homogeneity is %lf\n",h);
  
  return(h);
}

double CalculateCompleteness(vector<cluster> &GT, vector<cluster> &seg)
{
  double entropy_seg;
  entropy_seg = CalculateEntropy(seg);
  double C_entropy_seg_gt = CalculateConditionalEntropy(seg,GT);
  
 // double c = 1 - (C_entropy_seg_gt/entropy_seg);
  
  double c;
  
  if(C_entropy_seg_gt == 0)
    c = 1;
  else
    c =1 - (C_entropy_seg_gt/entropy_seg);
  
  
  printf("Completeness is %lf\n",c);
  
  return(c);
}

/*
 * @func: CalculateVmeasure : V measure, measures the harmonic mean of homogeneity and compactness
 * 
 * It can be used to evaluate aggrement of two independent assignment in same dataset
 * 
 * */

double CalculateVmeasure(vector<cluster> &GT, vector<cluster> &seg, int beta)
{
  double V;
  double h,c;
  h = CalculateHomogeneity(GT,seg);
  c = CalculateCompleteness(GT,seg);
  
  V = ((1 + beta) * (h * c))/((beta * h)+c);
  
  printf("V Measure is %lf\n",V);
  return(V);
  
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/



TDC Training(char *TrainFile)
{
  vector<FeatureVec> TrainData;
  vector<int> TrainClass;
  vector<int> ClassNumber;
  vector<int> NumberPerCluster;
  
  int classnumber = 0;
  
  FILE *f;
  f = fopen(TrainFile,"r");
  while(!feof(f))
  {
      
      char filename[2000];
      fscanf(f,"%s",&filename);
      printf("%s\n",filename);
      
      char *substring;
      char *groundtruthdest;
      //groundtruthdest = "stampVarOwnData/soumya/non_overlapped/Labelled/groundtruth";
      groundtruthdest = "300dpi/groundtruth";
      
      substring = input_image_name_cut(filename); 
      //makedir(substring);
      
      char *name,*output,*tempname;
  
  
      tempname = (char *) malloc ( 2001 * sizeof(char));
      if(tempname == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      tempname = CreateNameIntoFolder(groundtruthdest,substring);
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(tempname,"LabelAllInOne.png");
      
      Mat groundtruthdata = imread(name,0);
      
      
      
      printf("Binarization Done\n");
      
      vector<cluster> groundtruth_cluster;
      vector<int> groundtruth_hashtable;
      
	
      int hashnum = 0;
      
      for(int i=0;i<groundtruthdata.rows;i++)
      {
	for(int j=0;j<groundtruthdata.cols;j++)
	{
	  Point2i posi;
	  posi.x = j;
	  posi.y = i;
	  if(groundtruthdata.data[i*groundtruthdata.cols+j]!=255)
	  {
	    if(groundtruth_cluster.empty()) 
	    {
	      cluster x;
	      x.cluster_label = groundtruthdata.data[i*groundtruthdata.cols+j];
	      groundtruthdata.data[i*groundtruthdata.cols+j] = hashnum;
	      hashnum++;
	      x.p.push_back(posi);
	      groundtruth_cluster.push_back(x);
	      groundtruth_hashtable.push_back(x.cluster_label);
	      if(ClassNumber.empty())
	      {
		NumberPerCluster.push_back(0);
		ClassNumber.push_back(x.cluster_label);
		classnumber++;
	      }
	      else
	      {
		int m = 0;
		for(m=0;m<ClassNumber.size();m++)
		{
		  if(ClassNumber[m] == x.cluster_label)
		  {
		    break;
		  }
		}
		if(m == ClassNumber.size())
		{
		  NumberPerCluster.push_back(0);
		  ClassNumber.push_back(x.cluster_label);
		  classnumber++;
		}
	      }
	    }
	    else
	    {
	      // Find label;
	      int clabel = groundtruthdata.data[i*groundtruthdata.cols+j];
	      int k;
	      for(k=0;k<groundtruth_cluster.size();k++)
	      {
		if(groundtruth_cluster[k].cluster_label == clabel)
		{
		  int tr,tc;
		  tr = groundtruth_cluster[k].p[0].y;
		  tc = groundtruth_cluster[k].p[0].x;
		  groundtruthdata.data[i*groundtruthdata.cols+j] = groundtruthdata.data[tr*groundtruthdata.cols+tc];
		  groundtruth_cluster[k].p.push_back(posi);
		  break;
		}
	      }
	      if(k == groundtruth_cluster.size())
	      {
		cluster temp;
		temp.cluster_label = clabel;
		temp.p.push_back(posi);
		groundtruth_cluster.push_back(temp);
		groundtruth_hashtable.push_back(temp.cluster_label);
		groundtruthdata.data[i*groundtruthdata.cols+j] = hashnum;
		hashnum++;
		int m = 0;
		for(m=0;m<ClassNumber.size();m++)
		{
		  if(ClassNumber[m] == temp.cluster_label)
		  {
		    break;
		  }
		}
		if(m == ClassNumber.size())
		{
		  NumberPerCluster.push_back(0);
		  ClassNumber.push_back(temp.cluster_label);
		  classnumber++;
		}
	      }
	      
	    }
	  }
	}
      }
      
   //   printf("groundtruth cluster size %d\n",groundtruth_cluster.size());
      
   //   printf("%s\n",filename);
      
      char *inputdst;
    // inputdst = "600dpi/scans";
      inputdst = "300dpi/scans";
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(inputdst,filename);
      
      Mat GT = imread(name,1);
      
      int binarization_type; 
    // printf("Give Binarization Type :\n1 for adaptive\n2 for Otsu\n3 for binarization with GUI to select Threshold\n");
    // scanf("%d",&binarization_type);
      binarization_type = 4;
      
      Mat Gt_binary_dst = binarization(GT,binarization_type);
      
      Mat uniform_background;
  
      uniform_background = foreground_masked_image(GT,Gt_binary_dst);
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      name = CreateNameIntoFolder(substring,"uniform_background.png");
      //imwrite(name,uniform_background);
      
      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;
      
      Mat temp_img;
      
      Gt_binary_dst.copyTo(temp_img);
    // erosion_dst.copyTo(temp_img);
    // VGImage.copyTo(temp_img);
    // VGImage.release();
      
      temp_img = FindImageInverse(temp_img);
      
      /// Find contours
      findContours( temp_img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );	
      
      
      /// Approximate contours to polygons + get bounding rects and circles
      vector<vector<Point> > contours_poly( contours.size() );
      vector<Rect> boundRect( contours.size() );
      vector<RotatedRect> boundRotatedRect ( contours.size() );


      
      for( int j = 0; j < contours.size(); j++ )
	{ approxPolyDP( Mat(contours[j]), contours_poly[j], 3, true );
	  boundRect[j] = boundingRect( Mat(contours_poly[j]) );
	  boundRotatedRect[j] = minAreaRect( Mat(contours_poly[j]) );
	}
	
	
    /**
      * @Var initial_contour_flag
      * @brief size equal to number of original Contour from eroded image
      * @brief flag value is 1 if parent contour
      *        flag value is 0 if child contour
      *        flag value is 2 if contour are large_block(parent)
      *        flag value is 3 if contour ae small and suppose to be noise
      * */
      int *initial_contour_flag;
      initial_contour_flag = (int *)malloc(contours.size()*sizeof(int));
      for(int j = 0; j < contours.size(); j++ )
	initial_contour_flag[j]=1;
    
      for( int j = 0; j < contours.size(); j++ )
      {
	Rect TempRect;
	TempRect = boundRect[j];
	if(TempRect.width > GT.cols/4 || TempRect.height > GT.rows/4)
	{
	  initial_contour_flag[j]=2;
	}
      
      }
    Mat drawing;
    uniform_background.copyTo(drawing);
    Mat NewImage = Mat(GT.rows,GT.cols,GT.type(),Scalar(255,255,255));
    Mat NewBinaryDst = Mat(GT.rows,GT.cols,Gt_binary_dst.type(),Scalar(255));
    int number_of_block = 0;
    int p,q;
    for( int j = 0; j< contours.size(); j++ )
	{
	  if(initial_contour_flag[j]==1 && hierarchy[j][3] == -1)
	  {
	      //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	      //drawContours( drawing, contours_poly, j, color, 1, 8, vector<Vec4i>(), 0, Point() );
	      //rectangle( drawing, boundRect[j].tl(), boundRect[j].br(), color, 1, 8, 0 );
	      number_of_block = number_of_block + 1;
	      p = 0;
	      for(int m=boundRect[j].y;m<boundRect[j].y+boundRect[j].height;m++)
	      {
		q = 0;
		for(int n=boundRect[j].x;n<boundRect[j].x+boundRect[j].width;n++)
		{
		  int temp_col = boundRect[j].width;
		  bool measure_dist;
		  if((pointPolygonTest(contours_poly[j],Point(n,m),measure_dist) > 0.0) && Gt_binary_dst.data[m*Gt_binary_dst.cols+n]==0)
		  {
		    NewImage.at<Vec3b>(m,n)[0] = uniform_background.at<Vec3b>(m,n)[0];
		    NewImage.at<Vec3b>(m,n)[1] = uniform_background.at<Vec3b>(m,n)[1];
		    NewImage.at<Vec3b>(m,n)[2] = uniform_background.at<Vec3b>(m,n)[2];
		    NewBinaryDst.at<uchar>(m,n) = Gt_binary_dst.at<uchar>(m,n);
		  }
		}
	      }
	  }
	}
	
//       name = (char *) malloc ( 2001 * sizeof(char));
//       if(name == NULL)
//       {
// 	printf("Memory can not be allocated\n");
// 	exit(0);
//       }
//       name = CreateNameIntoFolder(substring,"contours.png");
//       imwrite(name,drawing);
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      name = CreateNameIntoFolder(substring,"NewImage.png");
      //imwrite(name,NewImage);
      
      //binarization_type = 3;
      //binary_dst = binarization(NewImage,binarization_type);
      
    // src.copyTo(NewImage);
    // binary_dst.copyTo(NewBinaryDst);
      
      
      Mat PrevBinaryDst;
      Gt_binary_dst.copyTo(PrevBinaryDst);
      Gt_binary_dst.release();
      NewBinaryDst.copyTo(Gt_binary_dst);
      NewBinaryDst.release();
      
      vector<Mat> hsv_planes;
      Mat hsv_image;
      //NewImage.convertTo(temp,CV_32FC3);
    //  cvtColor(NewImage,hsv_image,CV_BGR2HSV);
      //cvtColor(NewImage,hsv_image,CV_BGR2Lab);
      cvtColor(NewImage,hsv_image,CV_BGR2YCrCb);
      //hsv_image.convertTo(temp,CV_16UC3);
     split(hsv_image,hsv_planes);
      //split(NewHSVImage,hsv_planes);
     // split(NewImage, hsv_planes);
      
    /*
      for(i = 0; i < src.rows; i++)
      {
	for(j = 0; j < src.cols; j++)
	{
	  temp.at<Vec3b>(i,j)[0] = (hsv_image.at<Vec3b>(i,j)[0] * 2);
	}
      }
      split(temp,hsv_planes);
      */
      double temp_max_val;
      int temp_max_posi;
      
	int no_of_foregrnd_pix = NumberofForegroundPixel(Gt_binary_dst);
      
	
	for(int i=0;i<hsv_planes.size();i++)
	{
	  FindMaxElementPosi(hsv_planes[i],&temp_max_val,&temp_max_posi);
	  //printf("Max value of plane %d is %lf\n",i,temp_max_val);
	}

      doPCA(hsv_planes,Gt_binary_dst);
      
      //vector<float> colfeature = ScalarColorFeatureMasked(hsv_planes, Gt_binary_dst);
      Mat samples(no_of_foregrnd_pix, 1, CV_16UC1);
     
      float proj_val;
      Mat HSVProjectedData = Mat(NewImage.rows, NewImage.cols,CV_32FC1,Scalar(999));
      vector<float> temp_pdata;
      float max_pval = 0.0;
      float min_pval = 256.0;
      u_int16_t max_samval = 0;
      int k=0;
      for(int i = 0; i < GT.rows; i++)
      {
	for(int j = 0; j < GT.cols; j++)
	{
	  if(Gt_binary_dst.data[i*GT.cols+j] != 255)
	  {
	    proj_val = (h_unitvec * hsv_planes[0].at<uchar>(i,j) ) + ( s_unitvec * hsv_planes[1].at<uchar>(i,j) ) + ( v_unitvec * hsv_planes[2].at<uchar>(i,j) );
	    temp_pdata.push_back(proj_val);
	    if(min_pval >= proj_val)
	      min_pval = proj_val;
	    if(max_pval <= proj_val)
	      max_pval = proj_val;
	    //HSVProjectedData.at<u_int16_t>(i,j) =(u_int16_t) floor(proj_val);
	    //samples.at<u_int16_t>(k,0)= (u_int16_t) floor(proj_val);
	    //if(max_samval <= samples.at<u_int16_t>(k,0))
	      //max_samval = samples.at<u_int16_t>(k,0);
	    //k++;
	  }
	}
      }
      k = 0;
      for(int i = 0; i < GT.rows; i++)
      {
	for(int j = 0; j < GT.cols; j++)
	{
	  if(Gt_binary_dst.data[i*GT.cols+j] != 255)
	  {
	    if(min_pval < 0.0)
	    {
	      temp_pdata[k] = temp_pdata[k] - min_pval;
	    
	    }
	    HSVProjectedData.at<float>(i,j) = temp_pdata[k];
	    samples.at<u_int16_t>(k,0)= (u_int16_t) floor(temp_pdata[k]);
	    if(max_samval <= samples.at<u_int16_t>(k,0))
	      max_samval = samples.at<u_int16_t>(k,0);
	    k++;
	  }
	}
      }
      
      temp_pdata.clear();

      
    
      
      Mat HGImage = horizontal_gapfilling(Gt_binary_dst,25);
      Mat VGImage = vertical_gapfilling(HGImage,5);
    // namedWindow( "GapFilling", CV_WINDOW_KEEPRATIO ); 
    // imshow("GapFilling",VGImage);
    // waitKey(0);
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(substring,"GapFilledImage.png");
      //imwrite(name, VGImage);
      
      for( int j = 0; j< contours.size(); j++ )
      {
	contours[j].clear();
	contours_poly[j].clear();
      }
      contours.clear();
      contours_poly.clear();
      boundRect.clear();
      boundRotatedRect.clear();
      hierarchy.clear();
      
      if(contours.empty())
	printf("Contor is empty\n");
      
      temp_img.release();
      
      VGImage.copyTo(temp_img);
      VGImage.release();
      
      temp_img = FindImageInverse(temp_img);
      
      /// Find contours
      findContours( temp_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );	
      
     // printf("Contor Done\n");
      
      contours_poly.resize(contours.size());
      boundRect.resize(contours.size());
      boundRotatedRect.resize(contours.size());
      
      for( int j = 0; j < contours.size(); j++ )
      { 
	approxPolyDP( Mat(contours[j]), contours_poly[j], 3, true );
	boundRect[j] = boundingRect( Mat(contours_poly[j]) );
	boundRotatedRect[j] = minAreaRect( Mat(contours_poly[j]) );
      }
      
      
      initial_contour_flag = (int *)malloc(contours.size()*sizeof(int));
      for(int j = 0; j < contours.size(); j++ )
	initial_contour_flag[j]=1;
      
      GT.copyTo(drawing);
      number_of_block = 0;
      for( int j = 0; j< contours.size(); j++ )
      {
	if(hierarchy[j][3] == -1)
	{
	  Scalar color = Scalar( 0,0,0 );
	  rectangle( drawing, boundRect[j].tl(), boundRect[j].br(), color, 6, 8, 0 );
	  number_of_block = number_of_block + 1;
	} 
	else
	  initial_contour_flag[j] = 0;
      }
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(substring,"BoundingBox.png");
     // imwrite(name, drawing);
      
     // printf("Before Feature\n");
      
      vector<bool> validblock;
      //vector<double> BlockProjValMean;
      //vector<double> BlockProjValStdDev;
      //vector<double> BlockSWMean;
      //vector<double> BlockSWStdDev;
      //vector<int> Height;
      //vector<int> Width;
      //vector<Mat>Blocks;
      //vector<Mat>Binary_blocks;
      //vector<double> density;
      //vector<double> density1;
      //vector<Point2i> position;
      
     // printf("OK Upto here\n");
    //  printf("Contor size %d\n",contours.size());
      
      //&& groundtruthdata.data[m*groundtruthdata.cols+n]!=255
      
     // printf("groundtruth info row = %d col = %d\n",groundtruthdata.rows,groundtruthdata.cols);
     // printf("original info row = %d col = %d\n",Gt_binary_dst.rows,Gt_binary_dst.cols);
      
      for( int j = 0; j< contours.size(); j++ )
      {
	//printf("Block %d\n",j);
	validblock.push_back(false);
	
	if(hierarchy[j][3] == -1)
	{
	  
	//  printf("Block %d\n",j);
	  
	  
	  
	  Mat TempC = Mat(boundRect[j].height,boundRect[j].width,CV_8UC3,Scalar(255,255,255));
	  Mat TempB = Mat(boundRect[j].height,boundRect[j].width,CV_8UC1,Scalar(255));
	  
	  vector<float> ProjData;
	  int temp_pixel = 0;
	  int gtlabel;
	  p = 0;
	  
	  for(int m=boundRect[j].y;m<boundRect[j].y+boundRect[j].height;m++)
	  {
	    q=0;
	    for(int n=boundRect[j].x;n<boundRect[j].x+boundRect[j].width;n++)
	    {
	      
	      int temp_col = boundRect[j].width;
	      bool measure_dist;
	      if((pointPolygonTest(contours_poly[j],Point(n,m),measure_dist) > 0.0) && Gt_binary_dst.data[m*Gt_binary_dst.cols+n] == 0 && groundtruthdata.data[m*groundtruthdata.cols+n]!=255)
	      {
		
		temp_pixel = temp_pixel + 1;
		ProjData.push_back(HSVProjectedData.at<float>(m,n));
		//printf("%d\t%f\n",j,HSVProjectedData.at<float>(m,n));
		TempC.data[(p*temp_col+q)*3+0]=NewImage.data[(m*Gt_binary_dst.cols+n)*3+0];
		TempC.data[(p*temp_col+q)*3+1]=NewImage.data[(m*Gt_binary_dst.cols+n)*3+1];
		TempC.data[(p*temp_col+q)*3+2]=NewImage.data[(m*Gt_binary_dst.cols+n)*3+2];
		TempB.data[p*temp_col+q]=Gt_binary_dst.data[m*GT.cols+n];
		
		gtlabel = groundtruthdata.data[m*groundtruthdata.cols+n];
		//printf("gtlabel = %d\n",gtlabel);
		
	      }
	      
	      q++;
	    }
	    p++;
	  }
	  
	  
	  if(temp_pixel>2)
	  {
	    
	    vector<Ray> tempR = SWT(TempC,TempB);
	    if(tempR.size()>2)
	    {
	      
	     
	      vector<u_int16_t> TSW;
	      int i = 0;
	      for(vector<Ray>::iterator pid=tempR.begin();pid!=tempR.end();pid++)
	      {
		if(pid->dist > 2)
		{
		  TSW.push_back(pid->dist);
		  i = i + 1;
		}
	      }
	      
	      if(i > 2)
	      {
		// printf("Valid Block\n");
		validblock[j] = true;
		
		// Detecting Features for each valid blocks
		
		//Vec10f elem;
		FeatureVec elem;
		
		Mat tempstrokewidth = Mat(TSW.size(),1,CV_16UC1,TSW.data());
		elem[0] =(float) FindMean(tempstrokewidth);
		//BlockSWMean.push_back(FindMean(tempstrokewidth));
		elem[1] =(float) FindStdDev(tempstrokewidth);
		//BlockSWStdDev.push_back(FindStdDev(tempstrokewidth));
		tempstrokewidth.release();
		
		
		
		double tden = (temp_pixel/(boundRect[j].height*boundRect[j].width));
		//elem[4] =(float) tden; 
		//density.push_back(tden);
		tden = (temp_pixel)/(contourArea(contours[j]));
		//elem[5] =(float) tden;
		//density1.push_back(tden);
		
		//Height.push_back(boundRect[j].height);
	      // Width.push_back(boundRect[j].width);
		//elem[6] = boundRect[j].height *1.0;
		//elem[7] = boundRect[j].width * 1.0;
		
		//Blocks.push_back(TempC);
		//TempC.release();
		//Binary_blocks.push_back(TempB);
		//TempB.release();
		
		Mat ProjDataBlock = Mat(ProjData.size(),1,CV_32FC1,ProjData.data());
		ProjData.clear();
		
		//printf("Mean = %lf\t StdDev = %lf\n", FindMean(ProjDataBlock),FindStdDev(ProjDataBlock));
		
		elem[2] =(float) FindMean(ProjDataBlock);
	      // BlockProjValMean.push_back(FindMean(ProjDataBlock));
		//BlockProjValStdDev.push_back(FindStdDev(ProjDataBlock));
		elem[3] =(float) FindStdDev(ProjDataBlock);
		
		//printf("Mean = %f\t StdDev = %f\n", elem[2], elem[3]);
		
		ProjDataBlock.release();
		
		Point2i tposi;
		Rect trect = boundRect[j];
		tposi.x = trect.x + (trect.width/2);
		tposi.y = trect.y + (trect.height/2);
		//elem[8] = tposi.x * 1.0;
		//elem[9] = tposi.y * 1.0;
	      // position.push_back(tposi);
		
		//printf("gtlabel = %d\n",gtlabel);
		
		for(int l=0;l<ClassNumber.size();l++)
		{
		  if(ClassNumber[l] == groundtruth_hashtable[gtlabel])
		  {
		    NumberPerCluster[l] = NumberPerCluster[l] + 1;
		    if(NumberPerCluster[l] < 70)
		    {
		    
		      TrainData.push_back(elem);
		      TrainClass.push_back(groundtruth_hashtable[gtlabel]);
		      
		      
		      
		      groundtruth_cluster[gtlabel].blocks.push_back(j);
		      
		      //fprintf(train_features,"%f,%f,%f,%f,%s\n",elem[0],elem[1],elem[2],elem[3], GetClassName(groundtruth_hashtable[gtlabel]));
		    }
		    break;
		  }
		}
		
		//NumberPerCluster[ClassNumber[groundtruth_hashtable[gtlabel]]]++;
		
		
	      
	      }
	      
	    }
	    
	    
	    
	  }
	  
	  
	}
	
      }
      
     // printf("After Feature\n");
      
      
      
      
      
      
  }
  
  Mat trainSamples, trainClasses;
  
    Mat( TrainData ).copyTo( trainSamples );
    Mat( TrainClass ).copyTo( trainClasses );
    
    TrainData.clear();
    TrainClass.clear();

    
    printf("Sample Ori row=%d,col=%d,channels=%d\n",trainSamples.rows,trainSamples.cols,trainSamples.channels());
    // reshape trainData and change its type
    trainSamples = trainSamples.reshape( 1, trainSamples.rows );
    
    trainSamples.convertTo( trainSamples, CV_32FC1 );
    
    printf("Sample later row=%d,col=%d,channels=%d\n",trainSamples.rows,trainSamples.cols,trainSamples.channels());
  
    TDC Data;
    trainSamples.copyTo(Data.TrainData);
    trainClasses.copyTo(Data.TrainClass);
    Data.ClassNumber = ClassNumber;
    
    trainSamples.release();
    trainClasses.release();
    ClassNumber.clear();
  
    return(Data);
    
 
}

/*---------------------------------------------------------------------------------------------------------------------------------*/


/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./same_stamp <image-location- in text file> <color confidence> <strokewidth confidence>" << std::endl; }


/*-------------------------------------------------MAIN--------------------------------------------------------------------------------------*/



int main(int argc, char *argv[])
{
  
  TDC TrainedData = Training(argv[1]);
  
  vector<int> MulticlassTemp(TrainedData.ClassNumber.size(),0);
  MultiClassResSCCN.resize(TrainedData.ClassNumber.size(),MulticlassTemp);
  MultiClassResSCE.resize(TrainedData.ClassNumber.size(),MulticlassTemp);
  
  MulticlassTemp.clear();
  
  
  ConfusionMatrix C;
  C.initialize();
  vector<ConfusionMatrix> CM_ALL_SCCN(TrainedData.ClassNumber.size(),C);
  vector<ConfusionMatrix> CM_ALL_SCE(TrainedData.ClassNumber.size(),C);
  
  FILE *f_MC_SCCN,*f_MC_SCE;
  
  
  f_MC_SCCN = fopen("MulticlassCM_SCCN.xls","w");
  f_MC_SCE = fopen("MulticlassCM_SCE.xls","w");
  
  if( argc != 5 )
  { readme(); return -1; }
  
  FILE *f_svm_sccn; 
  f_svm_sccn = fopen("SCCN_SVM.xls","w");
  
  FILE *f_svm_sce; 
  f_svm_sce = fopen("SCE_SVM.xls","w");
  
  FILE *rinfo,*res;
  
  int cth = atoi(argv[3]);
  int sth = atoi(argv[4]);
  
  double ColorTh =(double) (cth*1.0)/100;
  double SWTh = (double) (sth*1.0)/100;
  
      char *filename;
      filename = (char *)malloc(3000 * sizeof(char));
      int vc = sprintf(filename,"Clustering_%d_%d.xls",cth,sth);
      rinfo = fopen(filename,"w");
      free(filename);
      filename = (char *)malloc(3000 * sizeof(char));
      vc = sprintf(filename,"Clustering_%d_%d.txt",cth,sth);
      res = fopen(filename,"w");
      free(filename);

 
  
   FILE *f;
  f = fopen(argv[2],"r");
 while(!feof(f))
 {
    //filename = (char *)malloc(2000*sizeof(char));
   char filename[2000];
    fscanf(f,"%s",&filename);
    printf("%s\n",filename);
  
  fprintf(rinfo,"%s",filename);
  
  
  
  
  
  //char *substring;
  
  Mat groundtruthdata;
  vector<int> groundtruth_hashtable;
  
  char *groundtruthdest;
  groundtruthdest = "300dpi/groundtruth";
  
  
 // substring = input_image_name_cut(argv[1]);
//  char *newfolder, *ou;
  
//   newfolder = (char *)malloc(3000 * sizeof(char));
//   newfolder = input_image_name_cut(filename);
//   ou = (char *)malloc(3000 * sizeof(char));
//   strcpy(ou,newfolder);
//   free(newfolder);
//   substring = (char *)malloc(3000 * sizeof(char));
//   strcpy(substring,ou);
//   char *tmpn;
//   tmpn = (char *)malloc(3000 * sizeof(char));
//   vc =  sprintf(tmpn,"_%d_%d",cth,sth);
//   printf("%s\n",tmpn);
//   strcat(substring,tmpn);
//   free(tmpn);
//   printf("%s\n",substring);
  
  
  substring = input_image_name_cut(filename); 
  makedir(substring);
  
  
  
  char *name,*output,*tempname;
  
  
  tempname = (char *) malloc ( 2001 * sizeof(char));
  if(tempname == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  
  tempname = CreateNameIntoFolder(groundtruthdest,substring);
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  
  name = CreateNameIntoFolder(tempname,"LabelAllInOne.png");
  
  groundtruthdata = imread(name,0);
  
  
  
  vector<cluster> groundtruth_cluster;
  int hashnum = 0;
  
  for(int i=0;i<groundtruthdata.rows;i++)
  {
    for(int j=0;j<groundtruthdata.cols;j++)
    {
      Point2i posi;
      posi.x = j;
      posi.y = i;
      if(groundtruthdata.data[i*groundtruthdata.cols+j]!=255)
      {
	if(groundtruth_cluster.empty()) 
	{
	  cluster x;
	  x.cluster_label = groundtruthdata.data[i*groundtruthdata.cols+j];
	  groundtruthdata.data[i*groundtruthdata.cols+j] = hashnum;
	  hashnum++;
	  x.p.push_back(posi);
	  groundtruth_cluster.push_back(x);
	  groundtruth_hashtable.push_back(x.cluster_label);
	}
	else
	{
	  // Find label;
	  int clabel = groundtruthdata.data[i*groundtruthdata.cols+j];
	  int k;
	  for(k=0;k<groundtruth_cluster.size();k++)
	  {
	    if(groundtruth_cluster[k].cluster_label == clabel)
	    {
	      int tr,tc;
	      tr = groundtruth_cluster[k].p[0].y;
	      tc = groundtruth_cluster[k].p[0].x;
	      groundtruthdata.data[i*groundtruthdata.cols+j] = groundtruthdata.data[tr*groundtruthdata.cols+tc];
	      groundtruth_cluster[k].p.push_back(posi);
	      break;
	    }
	  }
	  if(k == groundtruth_cluster.size())
	  {
	    cluster temp;
	    temp.cluster_label = clabel;
	    temp.p.push_back(posi);
	    groundtruth_cluster.push_back(temp);
	    groundtruth_hashtable.push_back(temp.cluster_label);
	    groundtruthdata.data[i*groundtruthdata.cols+j] = hashnum;
	    hashnum++;
	  }
	  
	}
      }
    }
  }
  
  printf("groundtruth cluster size %d\n",groundtruth_cluster.size());
  //imshow("test",groundtruthdata);
  //waitKey(0);
  
  char *inputdst;
    // inputdst = "600dpi/scans";
      inputdst = "300dpi/scans";
      
      
      // substring = input_image_name_cut(argv[1]);
      substring = input_image_name_cut(filename);
      //makedir(substring);
      
   // char *name,*tempname;
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(inputdst,filename);
      
      src = imread(name,1);
  int row,col;
  row = src.rows;
  col = src.cols;
  
   int i,j; 

  int binarization_type; 
 // printf("Give Binarization Type :\n1 for adaptive\n2 for Otsu\n3 for binarization with GUI to select Threshold\n");
 // scanf("%d",&binarization_type);
  binarization_type = 4;
   
  binary_dst = binarization(src,binarization_type);
  
  
  Mat boundary = boundaryextraction(binary_dst);
  
  /*
  for(int i=0;i<src.rows;i++)
  {
    for(int j=0;j<src.cols;j++)
    {
      if(binary_dst.at<uchar>(i,j) == 0 && boundary.at<uchar>(i,j) == 0)
      {
	binary_dst.at<uchar>(i,j) = 255;
      }
    }
  }
  boundary.release();
  */
  
  cvtColor(src,src_gray,CV_BGR2GRAY);
  //Mat Edge;
  //Canny(src_gray,Edge);
  //DrawHistogram(src_gray);
  //waitKey(0);
  Mat uniform_background;
  
  uniform_background = foreground_masked_image(src,binary_dst);
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"uniform_background.png");
 // imwrite(name,uniform_background);
  
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  
  Mat temp_img;
  
  binary_dst.copyTo(temp_img);
 // erosion_dst.copyTo(temp_img);
 // VGImage.copyTo(temp_img);
 // VGImage.release();
  
  temp_img = FindImageInverse(temp_img);
  
  /// Find contours
  findContours( temp_img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );	
  
  
  /// Approximate contours to polygons + get bounding rects and circles
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  vector<RotatedRect> boundRotatedRect ( contours.size() );


  
  for( int j = 0; j < contours.size(); j++ )
     { approxPolyDP( Mat(contours[j]), contours_poly[j], 3, true );
       boundRect[j] = boundingRect( Mat(contours_poly[j]) );
       boundRotatedRect[j] = minAreaRect( Mat(contours_poly[j]) );
     }
     
     
 /**
   * @Var initial_contour_flag
   * @brief size equal to number of original Contour from eroded image
   * @brief flag value is 1 if parent contour
   *        flag value is 0 if child contour
   *        flag value is 2 if contour are large_block(parent)
   *        flag value is 3 if contour ae small and suppose to be noise
   * */
  int *initial_contour_flag;
  initial_contour_flag = (int *)malloc(contours.size()*sizeof(int));
  for(int j = 0; j < contours.size(); j++ )
    initial_contour_flag[j]=1;
 
  for( int j = 0; j < contours.size(); j++ )
  {
    Rect TempRect;
    TempRect = boundRect[j];
    if(TempRect.width > src.cols/4 || TempRect.height > src.rows/4)
    {
      initial_contour_flag[j]=2;
    }
   
  }
 Mat drawing;
 uniform_background.copyTo(drawing);
 Mat NewImage = Mat(src.rows,src.cols,src.type(),Scalar(255,255,255));
 Mat NewBinaryDst = Mat(src.rows,src.cols,binary_dst.type(),Scalar(255));
 int number_of_block = 0;
 int p,q;
 for( int j = 0; j< contours.size(); j++ )
     {
       if(initial_contour_flag[j]==1 && hierarchy[j][3] == -1)
       {
	  Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	  //drawContours( drawing, contours_poly, j, color, 1, 8, vector<Vec4i>(), 0, Point() );
	  rectangle( drawing, boundRect[j].tl(), boundRect[j].br(), color, 1, 8, 0 );
	  number_of_block = number_of_block + 1;
	  p = 0;
	  for(int m=boundRect[j].y;m<boundRect[j].y+boundRect[j].height;m++)
	  {
	    q = 0;
	    for(int n=boundRect[j].x;n<boundRect[j].x+boundRect[j].width;n++)
	    {
	      int temp_col = boundRect[j].width;
	      bool measure_dist;
	      if((pointPolygonTest(contours_poly[j],Point(n,m),measure_dist) > 0.0) && binary_dst.data[m*binary_dst.cols+n]==0)
	      {
		NewImage.at<Vec3b>(m,n)[0] = uniform_background.at<Vec3b>(m,n)[0];
		NewImage.at<Vec3b>(m,n)[1] = uniform_background.at<Vec3b>(m,n)[1];
		NewImage.at<Vec3b>(m,n)[2] = uniform_background.at<Vec3b>(m,n)[2];
		NewBinaryDst.at<uchar>(m,n) = binary_dst.at<uchar>(m,n);
	      }
	    }
	  }
       }
     }
    
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"contours.png");
  //imwrite(name,drawing);
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"NewImage.png");
  //imwrite(name,NewImage);
  
  //binarization_type = 3;
  //binary_dst = binarization(NewImage,binarization_type);
  
 // src.copyTo(NewImage);
 // binary_dst.copyTo(NewBinaryDst);
  
  
  Mat PrevBinaryDst;
  binary_dst.copyTo(PrevBinaryDst);
  binary_dst.release();
  NewBinaryDst.copyTo(binary_dst);
  NewBinaryDst.release();
  
  vector<Mat> hsv_planes;
  Mat hsv_image;
  //NewImage.convertTo(temp,CV_32FC3);
//  cvtColor(NewImage,hsv_image,CV_BGR2HSV);
  //cvtColor(NewImage,hsv_image,CV_BGR2Lab);
  cvtColor(NewImage,hsv_image,CV_BGR2YCrCb);
  //hsv_image.convertTo(temp,CV_16UC3);
 // split(hsv_image,hsv_planes);
  //split(NewHSVImage,hsv_planes);
   split(NewImage, hsv_planes);
  
 /*
  for(i = 0; i < src.rows; i++)
  {
    for(j = 0; j < src.cols; j++)
    {
      temp.at<Vec3b>(i,j)[0] = (hsv_image.at<Vec3b>(i,j)[0] * 2);
    }
  }
  split(temp,hsv_planes);
  */
  double temp_max_val;
  int temp_max_posi;
  
    
  
    
    for(int i=0;i<hsv_planes.size();i++)
    {
      FindMaxElementPosi(hsv_planes[i],&temp_max_val,&temp_max_posi);
      printf("Max value of plane %d is %lf\n",i,temp_max_val);
    }

   
  doPCA(hsv_planes,binary_dst);
  Mat samples(no_of_foregrnd_pix, 1, CV_16UC1);
  float proj_val;
  Mat HSVProjectedData = Mat(NewImage.rows, NewImage.cols,CV_16UC1,Scalar(999));
  vector<float> temp_pdata;
  float max_pval = 0.0;
  float min_pval = 256.0;
  u_int16_t max_samval = 0;
  int k=0;
  for(i = 0; i < src.rows; i++)
  {
    for(j = 0; j < src.cols; j++)
    {
      if(binary_dst.data[i*src.cols+j] != 255)
      {
	proj_val = (h_unitvec * hsv_planes[0].at<uchar>(i,j) ) + ( s_unitvec * hsv_planes[1].at<uchar>(i,j) ) + ( v_unitvec * hsv_planes[2].at<uchar>(i,j) );
	temp_pdata.push_back(proj_val);
	if(min_pval >= proj_val)
	  min_pval = proj_val;
	if(max_pval <= proj_val)
	  max_pval = proj_val;
	//HSVProjectedData.at<u_int16_t>(i,j) =(u_int16_t) floor(proj_val);
	//samples.at<u_int16_t>(k,0)= (u_int16_t) floor(proj_val);
 	//if(max_samval <= samples.at<u_int16_t>(k,0))
 	  //max_samval = samples.at<u_int16_t>(k,0);
	//k++;
      }
    }
  }
  k = 0;
  for(i = 0; i < src.rows; i++)
  {
    for(j = 0; j < src.cols; j++)
    {
      if(binary_dst.data[i*src.cols+j] != 255)
      {
	if(min_pval < 0.0)
	{
	  temp_pdata[k] = temp_pdata[k] - min_pval;
	 
	}
	HSVProjectedData.at<u_int16_t>(i,j) =(u_int16_t) floor(temp_pdata[k]);
	samples.at<u_int16_t>(k,0)= (u_int16_t) floor(temp_pdata[k]);
	if(max_samval <= samples.at<u_int16_t>(k,0))
 	  max_samval = samples.at<u_int16_t>(k,0);
	k++;
      }
    }
  }
  
  temp_pdata.clear();
  
   printf("Max value calculated is %f and %d\n",max_pval,max_samval);
  
  max_samval = 0;
  for(int i=0;i<samples.rows;i++)
  {
    for(int j=0;j<samples.cols;j++)
    {
      if(max_samval <= samples.at<u_int16_t>(i,j))
	max_samval = samples.at<u_int16_t>(i,j);
    }
  }
  
  printf("Max value calculated is %f and %d\n",max_pval,max_samval);
  
  FindMaxElementPosi(samples,&temp_max_val,&temp_max_posi);
  printf("Max value %lf\n",temp_max_val);
  
  max(temp_max_val,samples);
  printf("Max value %lf\n",temp_max_val);
  
  Mat ProjDataHist = Mat(max_samval+1,1,CV_16UC1);
  for(int i=0;i<=max_samval;i++)
  {
    ProjDataHist.at<u_int16_t>(i,0) = 0;
  }
  for(int i=0;i<no_of_foregrnd_pix;i++)
  {
    u_int16_t temp_val = samples.at<u_int16_t>(i,0);
    ProjDataHist.at<u_int16_t>(temp_val,0) = ProjDataHist.at<u_int16_t>(temp_val,0) + 1;
  }
  
 Mat New_gray;
  cvtColor(NewImage,New_gray,CV_BGR2GRAY);
  
  Mat HGImage = horizontal_gapfilling(binary_dst,25);
  Mat VGImage = vertical_gapfilling(HGImage,5);
 // namedWindow( "GapFilling", CV_WINDOW_KEEPRATIO ); 
 // imshow("GapFilling",VGImage);
 // waitKey(0);
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  
  name = CreateNameIntoFolder(substring,"GapFilledImage.png");
  //imwrite(name, VGImage);
  
  for( int j = 0; j< contours.size(); j++ )
  {
    contours[j].clear();
    contours_poly[j].clear();
  }
  contours.clear();
  contours_poly.clear();
  boundRect.clear();
  boundRotatedRect.clear();
  hierarchy.clear();
  
  if(contours.empty())
    printf("Contor is empty\n");
  
  VGImage.copyTo(temp_img);
  VGImage.release();
  
  temp_img = FindImageInverse(temp_img);
  
  /// Find contours
  findContours( temp_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );	
  
  contours_poly.resize(contours.size());
  boundRect.resize(contours.size());
  boundRotatedRect.resize(contours.size());
  
  for( int j = 0; j < contours.size(); j++ )
  { 
    approxPolyDP( Mat(contours[j]), contours_poly[j], 3, true );
    boundRect[j] = boundingRect( Mat(contours_poly[j]) );
    boundRotatedRect[j] = minAreaRect( Mat(contours_poly[j]) );
  }
  
  
  initial_contour_flag = (int *)malloc(contours.size()*sizeof(int));
  for(int j = 0; j < contours.size(); j++ )
    initial_contour_flag[j]=1;
  
  src.copyTo(drawing);
  number_of_block = 0;
  for( int j = 0; j< contours.size(); j++ )
  {
    if(hierarchy[j][3] == -1)
    {
      Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
      rectangle( drawing, boundRect[j].tl(), boundRect[j].br(), color, 6, 8, 0 );
      number_of_block = number_of_block + 1;
    } 
    else
      initial_contour_flag[j] = 0;
  }
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  
  name = CreateNameIntoFolder(substring,"BoundingBox.png");
  //imwrite(name, drawing);
  
  vector<double> BlockProjValMean;
  vector<double> BlockProjValStdDev;
  vector<int> Height;
  vector<Mat> BlockProjectedData(number_of_block);
  vector<Mat>Blocks;
  vector<Mat>Binary_blocks;
  vector<int> BlockFP;
  nocc *ccblock;
  ccblock = (nocc *)malloc(number_of_block*sizeof(nocc));
  if(ccblock==NULL)
  {
    printf("Memory van not be allocated\n");
    exit(0);
  }
  
  
  int temp_pixel;
  double avg_val = 0;
  double avg_std_dev = 0;
  int avg_height = 0;
  k = 0;
  int l = 0;
  i = 0;
  for( int j = 0; j< contours.size(); j++ )
  {
    validblock.push_back(false);
    if(hierarchy[j][3] == -1)
    {
      
      ccblock[k].blk_no=k;
      ccblock[k].xmax= boundRect[j].x+boundRect[j].width;
      ccblock[k].ymax= boundRect[j].y+boundRect[j].height;
      ccblock[k].xmin= boundRect[j].x;
      ccblock[k].ymin= boundRect[j].y;
      ccblock[k].calc_centroid();
      ccblock[k].flag = 0;
      for(int v =0;v<256;v++)
      {
	ccblock[k].histogram[v]=0;
	ccblock[k].gray_hist[v]=0;
      }
      Mat TempC = Mat(boundRect[j].height,boundRect[j].width,CV_8UC3,Scalar(255,255,255));
      Mat TempB = Mat(boundRect[j].height,boundRect[j].width,CV_8UC1,Scalar(255));
      vector<u_int16_t> ProjData;
      temp_pixel = 0;
      p = 0;
      for(int m=boundRect[j].y;m<boundRect[j].y+boundRect[j].height;m++)
      {
	q=0;
	for(int n=boundRect[j].x;n<boundRect[j].x+boundRect[j].width;n++)
	{
	  int temp_col = boundRect[j].width;
	  bool measure_dist;
	  if((pointPolygonTest(contours_poly[j],Point(n,m),measure_dist) > 0.0) && binary_dst.data[m*binary_dst.cols+n]==0)
	  {
	    temp_pixel = temp_pixel + 1;
	    ProjData.push_back(HSVProjectedData.at<u_int16_t>(m,n));
	    int temp_gray_val = New_gray.data[m*New_gray.cols+n];
	    ccblock[k].gray_hist[temp_gray_val] = ccblock[k].gray_hist[temp_gray_val] + 1;
	    TempC.data[(p*temp_col+q)*3+0]=NewImage.data[(m*src_gray.cols+n)*3+0];
	    TempC.data[(p*temp_col+q)*3+1]=NewImage.data[(m*src_gray.cols+n)*3+1];
	    TempC.data[(p*temp_col+q)*3+2]=NewImage.data[(m*src_gray.cols+n)*3+2];
	    TempB.data[p*temp_col+q]=binary_dst.data[m*binary_dst.cols+n];
	  }
	  q++;
	}
	p++;
      }
      if(temp_pixel>2)
      {
	validblock[j] = true;
	Height.push_back(boundRect[j].height);
	avg_height = avg_height + boundRect[j].height;
	Mat ProjDataBlock = Mat(ProjData.size(),1,CV_16UC1,ProjData.data());
	ProjData.clear();
	BlockProjectedData[k] = ProjDataBlock;
	BlockProjValMean.push_back(FindMean(ProjDataBlock));
	avg_val = BlockProjValMean[l] + avg_val;
	BlockProjValStdDev.push_back(FindStdDev(ProjDataBlock));
	ProjDataBlock.release();
	avg_std_dev = avg_std_dev + BlockProjValStdDev[l];
	Blocks.push_back(TempC);
	TempC.release();
	Binary_blocks.push_back(TempB);
	TempB.release();
	BlockFP.push_back(temp_pixel);
	ccblock[k].flag = 1;
	ccblock[k].number_of_component = temp_pixel;
	ccblock[k].calc_distribution();
	l = l + 1;
      }    
      k = k + 1;
    }
  }
  avg_val = avg_val/l;
  avg_std_dev = avg_std_dev/l;
  avg_height = avg_height/l;
  
  int number_of_valid_block = l;
  
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"Blocks");
  //makedir(name);
  
  char *block_folder;
  block_folder = (char *) malloc ( 2001 * sizeof(char));
  strcpy(block_folder,name);
  
  int sp;
  
  // printing each block

  
  FILE *fp;
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"BlockWiseData.txt");
  fp = fopen(name, "w");
  
  
  vector<vector<int> > ccColor;
  vector<int> validblock_gc_table;
  vector<int> BlockPerClass(groundtruth_cluster.size());
  
  for(int i=0;i<groundtruth_cluster.size();i++)
  {
    BlockPerClass[i] = 0;
    vector<int> cl;
		for(int j=0;j<3;j++)
		  cl.push_back(rng.uniform(0, 255));
		ccColor.push_back(cl);
		cl.clear();
  }
  
  Mat Gimage;
  src.copyTo(Gimage);
  
  
 
  vector<vector<Ray> > strokes;
  vector<Mat> BlockSW;
  vector<double> BlockSWMean;
  vector<double> BlockSWStdDev;
  vector<double> Angle;
  vector<int> BlockSWSize;
  double avg_sw = 0.0;
  int no_strokes = 0;
  k = 0;
  l = 0;
  i = 0;
  int s = 0;
  for( int j = 0; j< contours.size(); j++ )
  {
    validblock[j] = false;
    if(hierarchy[j][3] == -1)
    {
      if(ccblock[k].flag == 1)
      {
	vector<Ray> tempR = SWT(Blocks[l],Binary_blocks[l]);
	if(tempR.size()>2)
	{
	  int gclabel =255;
	  for(int m=boundRect[j].y;m<boundRect[j].y+boundRect[j].height;m++)
	  {

	    for(int n=boundRect[j].x;n<boundRect[j].x+boundRect[j].width;n++)
	    {
	      if(binary_dst.data[m*binary_dst.cols+n]==0 && groundtruthdata.data[m*binary_dst.cols+n] != 255)
	      {		
			gclabel = groundtruthdata.data[m*groundtruthdata.cols+n];			
			BlockPerClass[gclabel] = BlockPerClass[gclabel] + 1;
			break;
	      }

	    }
	    if(gclabel != 255)
	    {
	      break;
	    }

	  }
	  if(gclabel != 255)
	  {
	    
	    vector<int> cl = ccColor[gclabel];
	    rectangle( Gimage, boundRect[j].tl(), boundRect[j].br(),Scalar(cl[0],cl[1],cl[2]), 6, 8, 0 );
	    cl.clear();
	    
	    validblock[j] = true;
	  //  printf("Gclabel = %d\n",gclabel);
	    validblock_gc_table.push_back(gclabel);
	    cluster tclus;
	    tclus = groundtruth_cluster[gclabel];
	    tclus.blocks.push_back(s);
	    groundtruth_cluster[gclabel] = tclus;
	  
	  
	 // if(gclabel!=255)
	    
	  
	 
	  //Mat tempstrokewidth = Mat(tempR.size(),1,CV_16UC1);
	    vector<u_int16_t> TSW;
	    i = 0;
	    for(vector<Ray>::iterator pid=tempR.begin();pid!=tempR.end();pid++)
	    {
	      if(pid->dist > 2)
	      {
		avg_sw = avg_sw + pid->dist;
		no_strokes = no_strokes + 1;
		//tempstrokewidth.at<u_int16_t>(i,0) = pid->dist;
		TSW.push_back(pid->dist);
		i = i + 1;
	      }
	    }
	    Mat tempstrokewidth = Mat(TSW.size(),1,CV_16UC1,TSW.data());
	    BlockSWSize.push_back(i);
	    BlockSW.push_back(tempstrokewidth);
	    BlockSWMean.push_back(FindMean(tempstrokewidth));
	    BlockSWStdDev.push_back(FindStdDev(tempstrokewidth));
	    tempstrokewidth.release();
	    RotatedRect RRectTemp;
	    RRectTemp = boundRotatedRect[j];
	    double temp_angle;
	    temp_angle =RRectTemp.angle;
	    if(temp_angle<=0)
	      temp_angle = 90 + temp_angle;
	    temp_angle = min(temp_angle,90-temp_angle);
	    Angle.push_back(temp_angle);
	  //  printf("For Block %d Proj Mean is %lf and dev is %lf SW Mean %lf and dev %lf and Angle %lf\n",l,BlockProjValMean[l],BlockProjValStdDev[l],BlockSWMean[s],BlockSWStdDev[s],Angle[s]);
	    fprintf(fp,"For Block %d Proj Mean is %lf and dev is %lf SW Mean %lf and dev %lf and Angle %lf\n",l,BlockProjValMean[l],BlockProjValStdDev[l],BlockSWMean[s],BlockSWStdDev[s],Angle[s]);
	    s = s + 1;
	  }
	}
	strokes.push_back(tempR);
	tempR.clear();
	l = l + 1;
      }
      k = k + 1;
    }
  }
  fclose(fp);
  avg_sw = avg_sw/no_strokes;

  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"GTImage.png");
  //imwrite(name,Gimage);
  Gimage.release();
  
  FILE *gtdata;
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"GTData.txt");
  gtdata = fopen(name, "w");
  
  for(int i=0;i<BlockPerClass.size();i++)
  {
    fprintf(gtdata,"%d\n",BlockPerClass[i]);
  }
  fclose(gtdata);\
  ccColor.clear();
  
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"ZValues.txt");
  fp = fopen(name,"w");
    
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"Center");
  //makedir(name);
  
  char *center;
  center = (char *) malloc ( 2001 * sizeof(char));
  strcpy(center,name);
  
    
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"Left");
  //makedir(name);
  
  char *left;
  left = (char *) malloc ( 2001 * sizeof(char));
  strcpy(left,name);
  
    
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"Right");
  //makedir(name);
  
  char *right;
  right = (char *) malloc ( 2001 * sizeof(char));
  strcpy(right,name);
  
    name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(center,"Other");
  //makedir(name);
  
  char *centerother;
  centerother = (char *) malloc ( 2001 * sizeof(char));
  strcpy(centerother,name);
  
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(left,"Other");
  //makedir(name);
  
  char *leftother;
  leftother = (char *) malloc ( 2001 * sizeof(char));
  strcpy(leftother,name);
  
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(right,"Other");
  //makedir(name);
  
  char *rightother;
  rightother = (char *) malloc ( 2001 * sizeof(char));
  strcpy(rightother,name);
  
  
  
  
  cluster cluster1_1,cluster1_2,cluster2_1,cluster2_2;
  
  Mat Class1 = Mat(src.rows,src.cols,src.type(),Scalar(255,255,255));
  Mat Class2 = Mat(src.rows,src.cols,src.type(),Scalar(255,255,255));
  Mat Class1_1 = Mat(src.rows,src.cols,src.type(),Scalar(255,255,255));
  Mat Class1_2 = Mat(src.rows,src.cols,src.type(),Scalar(255,255,255));
  Mat Class2_1 = Mat(src.rows,src.cols,src.type(),Scalar(255,255,255));
  Mat Class2_2 = Mat(src.rows,src.cols,src.type(),Scalar(255,255,255));
  
  uniform_background.copyTo(drawing);
  
  temp = Mat(BlockProjValMean.size(),1,CV_64FC1,BlockProjValMean.data());
  double MeanMean = FindMean(temp);
  double SEM = FindStdDev(temp);
  temp.release();
  temp = Mat(BlockSWMean.size(),1,CV_64FC1,BlockSWMean.data());
  double MeanSW = FindMean(temp);
  double SEM_SW = FindStdDev(temp);
  temp.release();
 //double MeanMean = FindMean(New_gray);
// double MeanMean = FindMean(samples);
  double StdDevMean = FindStdDev(samples);
  vector<double> ZValue;
  vector<double> PZValue;
  //vector<double> ProjPValue;
  vector<double> ZValueSW;
  vector<double> PZValueSW;
  vector<int> projcontournum;
  vector<int> SWcontournum;
  vector<int> SWProjnum;
  k = 0;
  l = 0;
  s = 0;
  for( int j = 0; j< contours.size(); j++ )
  {
    if(hierarchy[j][3] == -1)
    {
      if(ccblock[k].flag == 1)
      {
	projcontournum.push_back(j);
	//ZValue.push_back(CalculateZValue(BlockProjValMean[l],MeanMean,StdDevMean,ccblock[k].number_of_component));
	ZValue.push_back(CalculateZValue(BlockProjValMean[l],MeanMean,SEM));
	ae_int_t prosize = BlockFP[l];
	PZValue.push_back(studenttdistribution(prosize,ZValue[l]));
	double two_tail_pz,two_tail_pSW;
	  two_tail_pz = PZValue[l];
	  if(two_tail_pz > 0.5)
	    two_tail_pz = 2 * (1 - two_tail_pz);
	  else 
	    two_tail_pz = 2 * two_tail_pz;
	if(strokes[l].size() > 2 && validblock[j] == true)
	{
	  SWProjnum.push_back(l);
	  SWcontournum.push_back(j);
	  ZValueSW.push_back(CalculateZValue(BlockSWMean[s],MeanSW,SEM_SW));
	  ae_int_t swsize = BlockSW[s].rows;
	  PZValueSW.push_back(studenttdistribution(swsize,ZValueSW[s]));
	 // ProjPValue.push_back(PZValue[l]);
	  if(two_tail_pz > ColorTh)
	  {
	      two_tail_pSW = PZValueSW[s];
	      if(two_tail_pSW > 0.5)
		two_tail_pSW = 2 * (1 - two_tail_pSW);
	      else 
		two_tail_pSW = 2 * two_tail_pSW;
	      tempname = (char *) malloc ( 2001 * sizeof(char));
	      if(tempname == NULL)
	      {
		printf("Memory can not be allocated\n");
		exit(0);
	      }
	      sp =  sprintf(tempname,"block_%d.png",l);
	      name = (char *) malloc ( 2001 * sizeof(char));
	      if(name == NULL)
	      {
		printf("Memory can not be allocated\n");
		exit(0);
	      }  
	      name = CreateNameIntoFolder(center,tempname);
	      //imwrite(name,Blocks[l]);
	      
	      // writing image
	      
	      p = 0;
	      for(int m=boundRect[j].y;m<boundRect[j].y+boundRect[j].height;m++)
	      {
		q=0;
		for(int n=boundRect[j].x;n<boundRect[j].x+boundRect[j].width;n++)
		{
		  int temp_col = boundRect[j].width;
		  bool measure_dist;
		  if((pointPolygonTest(contours_poly[j],Point(n,m),measure_dist) > 0.0) && binary_dst.data[m*binary_dst.cols+n]==0)
		  {
		    Class1.at<Vec3b>(m,n)[0] = NewImage.at<Vec3b>(m,n)[0];
		    Class1.at<Vec3b>(m,n)[1] = NewImage.at<Vec3b>(m,n)[1];
		    Class1.at<Vec3b>(m,n)[2] = NewImage.at<Vec3b>(m,n)[2];
		  }
		  q++;
		}
		p++;
	      }
	      
	      if(two_tail_pSW > SWTh)
	      {
		
		cluster1_1.cluster_label = 0;
		cluster1_1.blocks.push_back(s);
		Point2i poi;
		rectangle( drawing, boundRect[j].tl(), boundRect[j].br(),Scalar(255,0,0), 6, 8, 0 );
		int pq = 0;
		for(int mn=boundRect[j].y;mn<boundRect[j].y+boundRect[j].height;mn++)
		{
		  poi.y = mn;
		  int qp=0;
		  for(int nm=boundRect[j].x;nm<boundRect[j].x+boundRect[j].width;nm++)
		  {
		    poi.x = nm;
		    int temp_col = boundRect[j].width;
		    bool measure_dist;
		    if((pointPolygonTest(contours_poly[j],Point(nm,mn),measure_dist) > 0.0) && binary_dst.data[mn*binary_dst.cols+nm]==0)
		    {
		      cluster1_1.p.push_back(poi);
		      Class1_1.at<Vec3b>(mn,nm)[0] = NewImage.at<Vec3b>(mn,nm)[0];
		      Class1_1.at<Vec3b>(mn,nm)[1] = NewImage.at<Vec3b>(mn,nm)[1];
		      Class1_1.at<Vec3b>(mn,nm)[2] = NewImage.at<Vec3b>(mn,nm)[2];
		    }
		    qp++;
		  }
		  pq++;
		}
	      }
	      else
	      {
		cluster1_2.cluster_label = 1;
		cluster1_2.blocks.push_back(s);
		rectangle( drawing, boundRect[j].tl(), boundRect[j].br(),Scalar(0,255,0), 6, 8, 0 );
		Point2i poi;
		int pq = 0;
		for(int mn=boundRect[j].y;mn<boundRect[j].y+boundRect[j].height;mn++)
		{
		  int qp=0;
		  poi.y = mn;
		  for(int nm=boundRect[j].x;nm<boundRect[j].x+boundRect[j].width;nm++)
		  {
		    poi.x = nm;
		    int temp_col = boundRect[j].width;
		    bool measure_dist;
		    if((pointPolygonTest(contours_poly[j],Point(nm,mn),measure_dist) > 0.0) && binary_dst.data[mn*binary_dst.cols+nm]==0)
		    {
		      cluster1_2.p.push_back(poi);
		      Class1_2.at<Vec3b>(mn,nm)[0] = NewImage.at<Vec3b>(mn,nm)[0];
		      Class1_2.at<Vec3b>(mn,nm)[1] = NewImage.at<Vec3b>(mn,nm)[1];
		      Class1_2.at<Vec3b>(mn,nm)[2] = NewImage.at<Vec3b>(mn,nm)[2];
		    }
		    qp++;
		  }
		  pq++;
		}
	      }
	      
	      
	  }
	  else
	  {
	      two_tail_pSW = PZValueSW[s];
	      if(two_tail_pSW > 0.5)
		two_tail_pSW = 2 * (1 - two_tail_pSW);
	      else 
		two_tail_pSW = 2 * two_tail_pSW;
	      
	      tempname = (char *) malloc ( 2001 * sizeof(char));
	      if(tempname == NULL)
	      {
		printf("Memory can not be allocated\n");
		exit(0);
	      }
	      sp =  sprintf(tempname,"block_%d.png",l);
	      name = (char *) malloc ( 2001 * sizeof(char));
	      if(name == NULL)
	      {
		printf("Memory can not be allocated\n");
		exit(0);
	      }  
	      name = CreateNameIntoFolder(centerother,tempname);
	      //imwrite(name,Blocks[l]);
	      
	      // drawing
	      
	      p = 0;
	      for(int m=boundRect[j].y;m<boundRect[j].y+boundRect[j].height;m++)
	      {
		q=0;
		for(int n=boundRect[j].x;n<boundRect[j].x+boundRect[j].width;n++)
		{
		  int temp_col = boundRect[j].width;
		  bool measure_dist;
		  if((pointPolygonTest(contours_poly[j],Point(n,m),measure_dist) > 0.0) && binary_dst.data[m*binary_dst.cols+n]==0)
		  {
		    Class2.at<Vec3b>(m,n)[0] = NewImage.at<Vec3b>(m,n)[0];
		    Class2.at<Vec3b>(m,n)[1] = NewImage.at<Vec3b>(m,n)[1];
		    Class2.at<Vec3b>(m,n)[2] = NewImage.at<Vec3b>(m,n)[2];
		  }
		  q++;
		}
		p++;
	      }
	      
	      
	      // 2nd class 
	      
	      if(two_tail_pSW > SWTh)
	      {
		cluster2_1.cluster_label = 2;
		cluster2_1.blocks.push_back(s);
		rectangle( drawing, boundRect[j].tl(), boundRect[j].br(),Scalar(0,0,255), 6, 8, 0 );
		Point2i poi;
		int pq = 0;
		for(int mn=boundRect[j].y;mn<boundRect[j].y+boundRect[j].height;mn++)
		{
		  poi.y = mn;
		  int qp=0;
		  for(int nm=boundRect[j].x;nm<boundRect[j].x+boundRect[j].width;nm++)
		  {
		    poi.x = nm;
		    int temp_col = boundRect[j].width;
		    bool measure_dist;
		    if((pointPolygonTest(contours_poly[j],Point(nm,mn),measure_dist) > 0.0) && binary_dst.data[mn*binary_dst.cols+nm]==0)
		    {
		      cluster2_1.p.push_back(poi);
		      Class2_1.at<Vec3b>(mn,nm)[0] = NewImage.at<Vec3b>(mn,nm)[0];
		      Class2_1.at<Vec3b>(mn,nm)[1] = NewImage.at<Vec3b>(mn,nm)[1];
		      Class2_1.at<Vec3b>(mn,nm)[2] = NewImage.at<Vec3b>(mn,nm)[2];
		    }
		    qp++;
		  }
		  pq++;
		}
	      }
	      else
	      {
		cluster2_2.cluster_label = 3;
		cluster2_2.blocks.push_back(s);
		Point2i poi;
		rectangle( drawing, boundRect[j].tl(), boundRect[j].br(),Scalar(120,150,120), 6, 8, 0 );
		int pq = 0;
		for(int mn=boundRect[j].y;mn<boundRect[j].y+boundRect[j].height;mn++)
		{
		  poi.y = mn;
		  int qp=0;
		  for(int nm=boundRect[j].x;nm<boundRect[j].x+boundRect[j].width;nm++)
		  {
		    poi.x = nm;
		    int temp_col = boundRect[j].width;
		    bool measure_dist;
		    if((pointPolygonTest(contours_poly[j],Point(nm,mn),measure_dist) > 0.0) && binary_dst.data[mn*binary_dst.cols+nm]==0)
		    {
		      cluster2_2.p.push_back(poi);
		      Class2_2.at<Vec3b>(mn,nm)[0] = NewImage.at<Vec3b>(mn,nm)[0];
		      Class2_2.at<Vec3b>(mn,nm)[1] = NewImage.at<Vec3b>(mn,nm)[1];
		      Class2_2.at<Vec3b>(mn,nm)[2] = NewImage.at<Vec3b>(mn,nm)[2];
		    }
		    qp++;
		  }
		  pq++;
		}
	      }
	      
	      
	  }
	  
	  
	  
	//  printf("Z value and P value for Block %d %d for PData is %lf and %lf\tZ value and P value of SW is %lf and %lf\n",l,s,ZValue[l],two_tail_pz,ZValueSW[s],two_tail_pSW);
	  fprintf(fp,"Z value for Block %d for PData is %lf\tZ value of SW is %lf\n",l,ZValue[l],ZValueSW[s]);
	  s = s + 1;
	}
	l = l + 1;
      }
      k = k + 1;
    }
  }
  fclose(fp);
  
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"class1.png");
  //imwrite(name,Class1);
  Class1.release();
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"class1_1.png");
 // imwrite(name,Class1_1);
  Class1_1.release();
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"class1_2.png");
 // imwrite(name,Class1_2);
  Class1_2.release();
  
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"class2.png");
 // imwrite(name,Class2);
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"class2_1.png");
 // imwrite(name,Class2_1);
  Class2_1.release();
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"class2_2.png");
 // imwrite(name,Class2_2);
  Class2_2.release();
  
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"all_class.png");
 // imwrite(name,drawing);
  drawing.release();
  
  
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"Ttest.xls");
  fp = fopen(name,"w");
  
  //printf("Before Finding Adjacency Matrix of Projected data\n");
  
  
  ZValueSW.clear();
  ZValue.clear();
  PZValue.clear();
  SWProjnum.clear();
  PZValueSW.clear();
  
 //  printf("hello\n");
  
  vector<cluster> SampleCluster;
  
  if(!cluster1_1.blocks.empty())
    SampleCluster.push_back(cluster1_1);
  if(!cluster1_2.blocks.empty())
    SampleCluster.push_back(cluster1_2);
  if(!cluster2_1.blocks.empty())
    SampleCluster.push_back(cluster2_1);
  if(!cluster2_2.blocks.empty())
    SampleCluster.push_back(cluster2_2);
  
  
  // printf("hello\n");
  
  vector<vector<int> > classperSampleCluster(SampleCluster.size());
  /*
  for(int i=0;i<validblock_gc_table.size();i++)
  {
	  printf("valid block %d label %d\n",i,validblock_gc_table[i]);
  }
  */
 // printf("hello\n");
  
 
  
  for(int i=0;i<SampleCluster.size();i++)
  {
      vector<int> x(groundtruth_cluster.size());
     for(int j=0;j<groundtruth_cluster.size();j++)
       x[j] = 0;
     for(int k=0;k<SampleCluster[i].blocks.size();k++)
     {
	//printf("clusternum %d\tblock no %d\tvalid blk %d\tval = %d\n",i,k,SampleCluster[i].blocks[k],validblock_gc_table[SampleCluster[i].blocks[k]]);
	x[validblock_gc_table[SampleCluster[i].blocks[k]]] = x[validblock_gc_table[SampleCluster[i].blocks[k]]] + 1;
     }
     classperSampleCluster[i] = x;
  }

  
  SampleCluster.clear();
  classperSampleCluster.clear();
  
  
  vector<FeatureVec> FV;
  
  Mat SWTtestVal = Mat(BlockSWMean.size(),BlockSWMean.size(),CV_64FC1,Scalar(255));
  //Mat SWEdgeGraph = Mat(BlockSWMean.size(),BlockSWMean.size(),CV_64FC1,Scalar(0));
  //vector<double> BlockTtest;
  Mat NewPTtestVal = Mat(BlockSWMean.size(),BlockSWMean.size(),CV_64FC1,Scalar(255));
  for(int i=0;i<BlockSWMean.size();i++)
  {
   // BlockTtest.push_back(oneSampleTtest(BlockSWMean[i],avg_sw,BlockSWStdDev[i],BlockSW[i].rows));
   // printf("TTest Val %d is %lf Mean %lf SM %lf\n",i,BlockTtest[i],avg_sw,BlockSWMean[i]);
    
    FeatureVec elem;
	elem[0] =(float) BlockSWMean[i];
	elem[1] =(float) BlockSWStdDev[i];
	elem[2] =(float) BlockProjValMean[SWProjnum[i]];
	elem[3] =(float) BlockProjValStdDev[SWProjnum[i]];
	
	FV.push_back(elem);
	
	
    
    for(int j=0;j<BlockSWMean.size();j++)
    {
      fprintf(fp,"\t%d",j);
    }
    fprintf(fp,"\n");
    fprintf(fp,"%d\t",i);
    for(int j=0;j<BlockSWMean.size();j++)
    {
      if(i!=j)
      {
	 
	
	
	SWTtestVal.at<double>(i,j) = twoSampleTtest(BlockSWSize[i],BlockSWSize[j],BlockSWMean[i],BlockSWMean[j],BlockSWStdDev[i],BlockSWStdDev[j]);
	SWTtestVal.at<double>(i,j) = studenttdistribution(BlockSWSize[i]+BlockSWSize[j],SWTtestVal.at<double>(i,j));
	if(SWTtestVal.at<double>(i,j) > 0.5)
	  SWTtestVal.at<double>(i,j) = 2 * (1 - SWTtestVal.at<double>(i,j));
	else
	  SWTtestVal.at<double>(i,j) = 2 * SWTtestVal.at<double>(i,j);
	//printf("%d %d %lf\n",i,j,TtestVal.at<double>(i,j));
	
	NewPTtestVal.at<double>(i,j) = twoSampleTtest(BlockFP[SWProjnum[i]],BlockFP[SWProjnum[j]],BlockProjValMean[SWProjnum[i]],BlockProjValMean[SWProjnum[j]],BlockProjValStdDev[SWProjnum[i]],BlockProjValStdDev[SWProjnum[j]]);
	//int dof =  
	NewPTtestVal.at<double>(i,j) = studenttdistribution(BlockFP[SWProjnum[i]]+BlockFP[SWProjnum[j]],NewPTtestVal.at<double>(i,j));
	if(NewPTtestVal.at<double>(i,j) > 0.5)
	 NewPTtestVal.at<double>(i,j) = 2 * (1 - NewPTtestVal.at<double>(i,j));
	else
	  NewPTtestVal.at<double>(i,j) = 2 * NewPTtestVal.at<double>(i,j);
      }
      fprintf(fp,"%lf\t",SWTtestVal.at<double>(i,j));
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
  
  name = (char *) malloc ( 2001 * sizeof(char));
  if(name == NULL)
  {
    printf("Memory can not be allocated\n");
    exit(0);
  }
  name = CreateNameIntoFolder(substring,"TtestProjVal.xls");
  fp = fopen(name,"w");
  
  Mat PTtestVal = Mat(BlockProjValMean.size(),BlockProjValMean.size(),CV_64FC1,Scalar(1000));
 // Mat PEdgeGraph = Mat(BlockProjValMean.size(),BlockProjValMean.size(),CV_8UC1,Scalar(0));
  for(int i=0;i<BlockProjValMean.size();i++)
  {
    for(int j=0;j<BlockProjValMean.size();j++)
    {
      fprintf(fp,"\t%d",j);
    }
    fprintf(fp,"\n");
    fprintf(fp,"%d\t",i);
    for(int j=0;j<BlockProjValMean.size();j++)
    {
      if(i != j)
      {
	PTtestVal.at<double>(i,j) = twoSampleTtest(BlockFP[i],BlockFP[j],BlockProjValMean[i],BlockProjValMean[j],BlockProjValStdDev[i],BlockProjValStdDev[j]);
	PTtestVal.at<double>(i,j) = studenttdistribution(BlockFP[i]+BlockFP[j],PTtestVal.at<double>(i,j));
	if(PTtestVal.at<double>(i,j) > 0.5)
	 PTtestVal.at<double>(i,j) = 2 * (1 - PTtestVal.at<double>(i,j));
	else
	  PTtestVal.at<double>(i,j) = 2 * PTtestVal.at<double>(i,j);
      }
      fprintf(fp,"%lf\t",PTtestVal.at<double>(i,j));
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
  
  
  PTtestVal.release();
  
  
  
  
  
  
  
  printf("Before Finding Adjacency Matrix of Projected data\n");
  
  
  
  vector<vector<bool> > ProjAdjMat;
 // ProjAdjMat = FindAdjacencyMatrix(PTtestVal, 0.05);
  ProjAdjMat = FindAdjacencyMatrix(NewPTtestVal, ColorTh);
  printf("After Finding Adjacency Matrix of Projected data\n");
  
  
  
  printf("Before Finding CC of Projected data\n");
  vector<vector<int> > Projcc;
  //vector<int> projlabels(number_of_valid_block);
  vector<int> projlabels(NewPTtestVal.rows);
  
  
  DFSCC(ProjAdjMat, Projcc, projlabels);
  
  ProjAdjMat.clear();
  printf("After Finding CC of Projected data\n");
  
  
  printf("Number of ConnectedComponent with Projected Data is %d\n",Projcc.size());
  
  
  printf("Before Finding Adjacency Matrix of SW data\n");
  vector<vector<bool> > SWAdjMat;
  SWAdjMat = FindAdjacencyMatrix(SWTtestVal, SWTh);
  printf("After Finding Adjacency Matrix of SW data\n");
  
  vector<vector<int> > SWcc;
  vector<int> SWlabels(SWTtestVal.rows);
  
  printf("Before Finding CC of SW data\n");
  
  DFSCC(SWAdjMat, SWcc, SWlabels);
  printf("After Finding CC of SW data\n");
  SWAdjMat.clear();
  
   printf("Number of ConnectedComponent with SW Data is %d\n",SWcc.size());
   
   
   name = (char *)malloc(2000*sizeof(char));
		name = CreateNameIntoFolder(substring, "HierarchycalCluster");
		//makedir(name);
		char *hierarchycluster;
		hierarchycluster = (char *)malloc(2000*sizeof(char));
		strcpy(hierarchycluster,name);
   
   vector<vector<int> > hierarchyCC;
   vector<int> hierarchylabels(SWTtestVal.rows);
   k = 0;
   for(int i=0;i<Projcc.size();i++)
   {
     vector<int> t1 = Projcc[i];
     for(int j=0;j<SWcc.size();j++)
     {
       vector<int> t3;
       
       vector<int> t2 = SWcc[j];
       t3 = FindIntersection(t1,t2);
       if(!t3.empty()) // if intersection is not empty create new cluster
       {
	 for(int m=0;m<t3.size();m++)
	   hierarchylabels[t3[m]] = k;
	 hierarchyCC.push_back(t3);
	 k = k + 1;
       }
       t3.clear();
       
       t2.clear();
     }
     t1.clear();
   }
   
      	    
	 
	      

	      
  vector<vector<bool> > AdjMat;

  double th1 = ColorTh;
  double th2 = SWTh; 
  AdjMat = FindAdjacencyMatrix2data(NewPTtestVal, th1, SWTtestVal, th2);
  vector<vector<int> > adjcc;
  vector<int> adjlabels(NewPTtestVal.rows);
  DFSCC(AdjMat, adjcc, adjlabels);
  
  vector<int> gtlabel(validblock_gc_table.size());
  
  for(int i=0;i<validblock_gc_table.size();i++)
  {
    gtlabel[i] = groundtruth_hashtable[validblock_gc_table[i]];
  }
  
   
#if _SVM_
    
    
	    
	    vector<ConfusionMatrix> CM_SVM_SCCN = segmentation_withSVM(FV, gtlabel, hierarchylabels, TrainedData, 1);
	    
	    printf("SVM Done\n");
	    
	    
	    
	    ConfusionMatrix overall;
	    overall.initialize();
	    
	    
	    fprintf(f_svm_sccn,"%s\t",substring);
	    for(int i=0;i<CM_SVM_SCCN.size();i++)
	    {
	      ConfusionMatrix cm_i = CM_SVM_SCCN[i];
	      fprintf(f_svm_sccn,"%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",TrainedData.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
	     // printf("%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
	      overall.tp = overall.tp + cm_i.tp;
	      overall.fp = overall.fp + cm_i.fp;
	      overall.tn = overall.tn + cm_i.tn;
	      overall.fn = overall.fn + cm_i.fn;
	      
	      
	      CM_ALL_SCCN[i].tp = CM_ALL_SCCN[i].tp + cm_i.tp;
	      CM_ALL_SCCN[i].fp = CM_ALL_SCCN[i].fp + cm_i.fp;
	      CM_ALL_SCCN[i].tn = CM_ALL_SCCN[i].tn + cm_i.tn;
	      CM_ALL_SCCN[i].fn = CM_ALL_SCCN[i].fn + cm_i.fn;

	      
	      

	    }
	    fprintf(f_svm_sccn,"overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
	    //printf("overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
	    fprintf(f_svm_sccn,"\n");
	    
	    
	    CM_SVM_SCCN.clear();
	    
	    
	    
	    
	    
	    
	    vector<ConfusionMatrix> CM_SVM_SCE = segmentation_withSVM(FV, gtlabel, adjlabels, TrainedData, 2);
	    
	    printf("SVM Done\n");
	    

	    
	    
	    
	    overall.initialize();
	    
	    
	    fprintf(f_svm_sce,"%s\t",substring);
	    for(int i=0;i<CM_SVM_SCE.size();i++)
	    {
	      ConfusionMatrix cm_i = CM_SVM_SCE[i];
	      fprintf(f_svm_sce,"%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",TrainedData.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
	     // printf("%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
	      overall.tp = overall.tp + cm_i.tp;
	      overall.fp = overall.fp + cm_i.fp;
	      overall.tn = overall.tn + cm_i.tn;
	      overall.fn = overall.fn + cm_i.fn;
	      
	      CM_ALL_SCE[i].tp = CM_ALL_SCE[i].tp + cm_i.tp;
	      CM_ALL_SCE[i].fp = CM_ALL_SCE[i].fp + cm_i.fp;
	      CM_ALL_SCE[i].tn = CM_ALL_SCE[i].tn + cm_i.tn;
	      CM_ALL_SCE[i].fn = CM_ALL_SCE[i].fn + cm_i.fn;
	      
	      

	    }
	    fprintf(f_svm_sce,"overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
	    //printf("overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
	    fprintf(f_svm_sce,"\n");
	    
	    
	    
	    
    
#endif
  
	
	
  
  NewPTtestVal.release();
  SWTtestVal.release();
  
  
  groundtruth_cluster.clear();
  groundtruth_hashtable.clear();
  groundtruthdata.release();
  validblock_gc_table.clear();
  
  
  fprintf(rinfo,"\n");
  fprintf(res,"\n");
  
  
 }
  
  for(int i=0;i<MultiClassResSCCN.size();i++)
  {
    vector<int> MC_i = MultiClassResSCCN[i];
    for(int j=0;j<MC_i.size();j++)
    {
      fprintf(f_MC_SCCN,"%d\t",MC_i[j]);
      printf("%d\t",MC_i[j]);
    }
    fprintf(f_MC_SCCN,"\n");
    printf("\n");
  }
  fclose(f_MC_SCCN);
  
  for(int i=0;i<MultiClassResSCE.size();i++)
  {
    vector<int> MC_i = MultiClassResSCE[i];
    for(int j=0;j<MC_i.size();j++)
    {
      fprintf(f_MC_SCE,"%d\t",MC_i[j]);
      printf("%d\t",MC_i[j]);
    }
    fprintf(f_MC_SCE,"\n");
    printf("\n");
  }
  fclose(f_MC_SCE);
 
  
  fclose(rinfo);
  fclose(res);
  
  fclose(f_svm_sccn);
  fclose(f_svm_sce);
  
  
  
 FILE *overallclassres;
 
 overallclassres = fopen("OverAllClassificationResult_Clustering.xls","w");
 
 fprintf(overallclassres,"ALgo/Metrics\tAverageAccuracy\tErrorRate\tprecession_{\mu}\trecall_{\mu}\tfscore_{\mu}\tprecession_{M}\trecall_{M}\tfscore_{M}\n");
 
 
 MultiClassPerformanceMetrics M;
  
  M.initialize(CM_ALL_SCCN);
 
  fprintf(overallclassres,"SCCN\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",M.GetAverageAccuracy(),M.GetErrorRate(),M.GetPrecesionMu(),M.GetRecallMu(),M.GetFScoreMu(1),M.GetPrecesionM(),M.GetRecallM(),M.GetFScoreM(1));
 
  M.Clear();
  CM_ALL_SCCN.clear();
  
  
  M.initialize(CM_ALL_SCE);
 
  fprintf(overallclassres,"SCE\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",M.GetAverageAccuracy(),M.GetErrorRate(),M.GetPrecesionMu(),M.GetRecallMu(),M.GetFScoreMu(1),M.GetPrecesionM(),M.GetRecallM(),M.GetFScoreM(1));
 
  M.Clear();
  CM_ALL_SCE.clear();
  
  
  return 0;
}
