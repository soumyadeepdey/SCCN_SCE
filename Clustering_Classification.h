#include "StandardHeaders.h"

#include "binarization.h"
#include "folder.h"
#include "Image_proc_functions.h"
#include "ScalarColorFeature.h"
#include "StatisticalFunctions.h"
#include "SmoothingGapfilling.h"
#include "StrokeWidth.h"




using namespace IITkgp_functions;


#include "alglib/ap.h"
#include "alglib/alglibinternal.h"
#include "alglib/alglibmisc.h"
#include "alglib/linalg.h"
#include "alglib/specialfunctions.h"
#include "alglib/statistics.h"


using namespace alglib;



#ifdef HAVE_OPENCV_OCL
#define _OCL_KNN_ 0 // select whether using ocl::KNN method or not, default is using
#define _OCL_SVM_ 0 // select whether using ocl::svm method or not, default is using
#include "opencv2/ocl/ocl.hpp"
#endif




#define _NBC_ 1 // normal Bayessian classifier
#define _KNN_ 1 // k nearest neighbors classifier
#define _SVM_ 1 // support vectors machine
#define _DT_  1 // decision tree
#define _BT_  0 // ADA Boost
#define _GBT_ 0 // gradient boosted trees
#define _RF_  1 // random forest
#define _ERT_ 0 // extremely randomized trees
#define _ANN_ 0 // artificial neural networks
#define _EM_  1 // expectation-maximization


bool SCCN_flag = false;
bool SCE_flag = false;

//FILE *sw_feature;
//FILE *train_features;
//FILE *test_features;

//int n = 4;

const int vecsize = 4;

typedef Vec<float, vecsize> FeatureVec;

//typedef Vec<float, 10> Vec10f;

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

typedef struct TrainDataClass
{
  Mat TrainData;
  Mat TrainClass;
  vector<int> ClassNumber;
}TDC;



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
    
    void SetSWSize(int val)
    {
      SWSize = val;
    }
    
    int GetColorBlkSize()
    {
      SetColorBlkSize();
      return(ColorBlkSize);
    }
    
    int GetSWSize()
    {
      return(SWSize);
    }
    
    void Destroy()
    {
      BlockContour.clear();
      BlockPoly.clear();
      ColorBlock.release();
      BinaryBlock.release();
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
    int SWSize;
    int ColorBlkSize;
    
   
     void SetColorBlkSize()
    {
      ColorBlkSize = NumberofForegroundPixel(BinaryBlock);
    }
    
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

typedef struct ValidBlockandFeature
{
  vector<FeatureVec> Features;
  vector<bool> ValidBlocks;
}VBF;

typedef struct ClusterDeatils
{
  int cluster_label; 
  vector<Point2i> p;
  vector<int> blocks;
}cluster;

typedef struct ClusterInformation
{
  vector<vector<int> > clusters;
  vector<int> clusternum;
}clusinfo;



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

/**************************************************FUNCTIONS****************************************************/


void PrintClassName(int label);

char* GetClassName(int label);

bool labelisgraphics(int label);

bool labelisstamp(int label);

bool labelislogo(int label);

bool labelistext(int label);

bool labelisboldtext(int label);

bool labelisheader(int label);

Mat GetPreprocessedImageForProcessingBlock(Mat Image);

vector<ProcessingBlock> GetProcessingBlocks(Mat Image);

vector<ProcessingBlock> GetProcessingBlocksWithFeatures(Mat Image);


/**
 * @function FindAdjacencyMatrix
 * @param : input : vector<vector<type> > - A adjacency matrix with relation value between two node is given, in double
 * 		    double connection_condition - It is the value by checking which two nodes are connected
 * 
 * example : connection_condition = x;
 * 	     if(RelationMat.at<double>(i,j) > x)
 * 		
 * return : A AdjacencyMatrix of bool type (vector<vector<bool> >)
 * */

template <typename T>
vector<vector<bool> > FindAdjacencyMatrix(vector<vector<T> > &Relation, T connection_condition);


template <typename T>
vector<vector<bool> > FindAdjacencyMatrix2data(vector<vector<T> > &Relation1, T connection_condition1, vector<vector<T> > &Relation2, T connection_condition2);


void GiveLabelDFS(vector<vector<bool> > AdjMat, vector<int> &tempcomponent, vector<int> &labels, vector<bool> &ccflag);


/**
 * @func : 		DFSCC 
 * @desc :		Find ConnectedComponent based on depth First search of the adjacency matrix
 * @param:	input:	vector<vector<bool> > AdjMat :	AdjacencyMatrix in bool format
 * 
 * 			vector<vector<int> > &component: address to the component
 * 
 * */

void DFSCC(vector<vector<bool> > AdjMat, vector<vector<int> > &component, vector<int> &labels);


template <typename T>
vector<T> FindIntersection(vector<T> &v1, vector<T> &v2);

double CalculateZValue(double SampleMean, double PopulationMean, double PopulationSD, int SampleSize);

double CalculateZValue(double SampleMean, double PopulationMean, double SEM);



double oneSampleTtest(double sample_mean, double population_mean, double sample_sd, double sample_size);

double twoSampleTtest_UnequalVariance(int sample1_size, int sample2_size, double sample1_mean, double sample2_mean, double sample1_sd, double sample2_sd);

double twoSampleTtest_EqualVariance(int sample1_size, int sample2_size, double sample1_mean, double sample2_mean, double sample1_sd, double sample2_sd);

int DegreeofFreedom_UnequalVariance(int sample1_size, int sample2_size, double sample1_sd, double sample2_sd);

int DegreeofFreedom_EqualVariance(int sample1_size, int sample2_size);

clusinfo ClusteringSCCN(vector<ProcessingBlock> &PB, double ColorTh, double SWTh);

clusinfo ClusteringSCE(vector<ProcessingBlock> &PB, double ColorTh, double SWTh);

TDC Training(char *TrainFile);

/**
 * @func GetConfusionMatrix
 * @param vector<ProcessingBlock> : Send ProcessingBlock after Prediction
 * @param TDC : Send the Training Data Class
 * @return vector<ConfusionMatrix> : Return Confution matrix for the Processing blocks
 * */
vector<ConfusionMatrix> GetConfusionMatrix(vector<ProcessingBlock> PB, vector<int> ClassNumber);


void segmentation_withKNN(vector<ProcessingBlock> &PB, TDC Data, int K);

void segmentation_withNBC(vector<ProcessingBlock> &PB, TDC Data);

void segmentation_withEM(vector<ProcessingBlock> &PB, TDC Data);

void segmentation_withSVM(vector<ProcessingBlock> &PB, TDC Data);

void segmentation_withDT(vector<ProcessingBlock> &PB, TDC Data);

void segmentation_withRF(vector<ProcessingBlock> &PB, TDC Data);



void classification(vector<ProcessingBlock> PB, TDC Data);



template <typename T>
int FindPosition(vector<T> x, T val);


void PBLabelsWRTClusterLabels(clusinfo Algo, vector<ProcessingBlock> &PB, vector<int> ClassNumber);

void Clustering_Classification(clusinfo Algo, vector<ProcessingBlock> &PB, TDC Data, char *ClassifierName);






