



#include "Clustering_Classification.h"


 
#if _NBC_  
  FILE *f_nbc;
#endif 
  
#if _EM_
  FILE *f_em; 
#endif  
  
#if _KNN_
  int K;
  FILE *f_knn;
#endif
  
#if _SVM_
  FILE *f_svm;
#endif
  
#if _DT_
  FILE *f_dt;
#endif
  
#if _RF_
  FILE *f_rf;
#endif


char *substring;
/*-------------------------------------------------------------------------------------------------------------------------------------------*/

void PrintClassName(int label)
{
  if(label == 0)
    printf("Text\n");
  if(label == 1)
    printf("BoldText\n");
  if(label == 2)
    printf("Graphics\n");
  if(label == 3)
    printf("Logo\n");
  if(label == 4)
    printf("Stamp\n");
  if(label == 5)
    printf("Headers\n");
  if(label == 6)
    printf("HeadLine\n");
  if(label == 7)
    printf("Signature\n");
  if(label == 8)
    printf("Noise\n"); 
}


char* GetClassName(int label)
{
  char *name;
  name = (char *)malloc(2000*sizeof(char));
  if(label == 0)
    name = "Text";
  else if(label == 1)
   name = "BoldText";
  else if(label == 2)
    name = "Graphics";
  else if(label == 3)
    name = "Logo";
  else if(label == 4)
    name  = "Stamp";
  else if(label == 5)
    name = "Headers";
  else if(label == 6)
    name = "HeadLine";
  else if(label == 7)
    name = "Signature";
  else if(label == 8)
    name = "Noise";
  return(name);
  
}

bool labelisgraphics(int label)
{
  bool t;
  if(label == 2)
    t = true;
  else
    t = false;
  return t;
}

bool labelisstamp(int label)
{
  bool t;
  if(label == 4)
    t = true;
  else
    t = false;
  return t;
}

bool labelislogo(int label)
{
  bool t;
  if(label == 3)
    t = true;
  else
    t = false;
  return t;
}

bool labelistext(int label)
{
  bool t;
  if(label == 0)
    t = true;
  else
    t = false;
  return t;
}

bool labelisboldtext(int label)
{
  bool t;
  if(label == 1)
    t = true;
  else
    t = false;
  return t;
}

bool labelisheader(int label)
{
  bool t;
  if(label == 5)
    t = true;
  else
    t = false;
  return t;
}





/*-------------------------------------------------------------------------------------------------------------------------------------------*/


Mat GetPreprocessedImageForProcessingBlock(Mat Image)
{
   Mat GT;
      Image.copyTo(GT);
      Image.release();
      
      int binarization_type; 
    // printf("Give Binarization Type :\n1 for adaptive\n2 for Otsu\n3 for binarization with GUI to select Threshold\n");
    // scanf("%d",&binarization_type);
      binarization_type = 5;
      
      Mat Gt_binary_dst = binarization(GT,binarization_type);
      
      Mat uniform_background;
  
      uniform_background = foreground_masked_image(GT,Gt_binary_dst);
      
      
      
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
	

      
      return(NewImage);
      
      
}

vector<ProcessingBlock> GetProcessingBlocks(Mat Image)
{
  
     
  
      int binarization_type; 
    // printf("Give Binarization Type :\n1 for adaptive\n2 for Otsu\n3 for binarization with GUI to select Threshold\n");
    // scanf("%d",&binarization_type);
      binarization_type = 5;
      
      Mat Gt_binary_dst = binarization(Image,binarization_type);
      
     
      Mat HGImage = horizontal_gapfilling(Gt_binary_dst,32);
      Mat VGImage = vertical_gapfilling(HGImage,5);
      
      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;

      Mat temp_img;
      VGImage.copyTo(temp_img);
      VGImage.release();
      HGImage.release();
      
      temp_img = FindImageInverse(temp_img);
      
      /// Find contours
      findContours( temp_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );	
      
      vector<vector<Point> > contours_poly( contours.size() );
      vector<Rect> boundRect( contours.size() );
      
      for( int j = 0; j < contours.size(); j++ )
      { 
	approxPolyDP( Mat(contours[j]), contours_poly[j], 3, true );
	boundRect[j] = boundingRect( Mat(contours_poly[j]) );
      }

    
      
      vector<ProcessingBlock> ValidPB;
      int p,q;
      for( int j = 0; j< contours.size(); j++ )
      {
	
	if(hierarchy[j][3] == -1)
	{
	  ProcessingBlock PB;
	  PB.initialize();
	  Mat TempC = Mat(boundRect[j].height,boundRect[j].width,CV_8UC3,Scalar(255,255,255));
	  Mat TempB = Mat(boundRect[j].height,boundRect[j].width,CV_8UC1,Scalar(255));
	  
	  int temp_pixel = 0;
	 
	  p = 0;
	  for(int m=boundRect[j].y;m<boundRect[j].y+boundRect[j].height;m++)
	  {
	    q=0;
	    for(int n=boundRect[j].x;n<boundRect[j].x+boundRect[j].width;n++)
	    {
	      int temp_col = boundRect[j].width;
	      bool measure_dist;
	      if((pointPolygonTest(contours_poly[j],Point(n,m),measure_dist) > 0.0) && Gt_binary_dst.data[m*Gt_binary_dst.cols+n]==0)
	      {
		temp_pixel = temp_pixel + 1;
		TempC.data[(p*temp_col+q)*3+0]=Image.data[(m*Image.cols+n)*3+0];
		TempC.data[(p*temp_col+q)*3+1]=Image.data[(m*Image.cols+n)*3+1];
		TempC.data[(p*temp_col+q)*3+2]=Image.data[(m*Image.cols+n)*3+2];
		TempB.data[p*temp_col+q]=Gt_binary_dst.data[m*Image.cols+n];
		
	      }
	      q++;
	    }
	    p++;
	  }
	  
	  if(temp_pixel > 2)
	  {
	    PB.ColorBlock = TempC;
	    PB.BinaryBlock = TempB;
	    PB.BlockContour = contours[j];
	    PB.valid = true;
	    
	    ValidPB.push_back(PB);
	  }
	  	  
	}
      }
      
      contours.clear();
      boundRect.clear();

      
      return(ValidPB);
}




vector<ProcessingBlock> GetProcessingBlocksWithFeatures(Mat Image)
{
      printf("In ProcessingBlock \n");
      Image = GetPreprocessedImageForProcessingBlock(Image);
      
      vector<ProcessingBlock> PB = GetProcessingBlocks(Image);
      
      vector<Mat> color_planes;
      Mat color_image;
     
    //  cvtColor(Image,color_image,CV_BGR2HSV);
      //cvtColor(Image,color_image,CV_BGR2Lab);
      cvtColor(Image,color_image,CV_BGR2YCrCb); 
    // split(color_image,color_planes);
      split(color_image, color_planes);
      
      int bin_type = 5;
      
      Mat Binary = binarization(Image, bin_type);
      
      vector<float> colfeature = ScalarColorFeatureMasked(color_planes, Binary);
      
    //  vector<Vec10f> features;
      
      
      for( int m = 0; m< PB.size(); m++ )
      {
	ProcessingBlock pb_j = PB[m];
	Mat CImg;
	cvtColor(pb_j.ColorBlock,CImg,CV_BGR2YCrCb); 
	float proj_val;
	vector<float> temp_pdata;
	float max_pval = 0.0;
	float min_pval = 256.0;
	u_int16_t max_samval = 0;
	int k=0;
	for(int i = 0; i < pb_j.ColorBlock.rows; i++)
	{
	  for(int j = 0; j < pb_j.ColorBlock.cols; j++)
	  {
	    if(pb_j.BinaryBlock.data[i*pb_j.BinaryBlock.cols+j] != 255)
	    {
	      proj_val = (colfeature[0] * CImg.at<Vec3b>(i,j)[0] ) + ( colfeature[1] * CImg.at<Vec3b>(i,j)[1] ) + ( colfeature[2] * CImg.at<Vec3b>(i,j)[2] );
	      temp_pdata.push_back(proj_val);
	      if(min_pval >= proj_val)
		min_pval = proj_val;
	      if(max_pval <= proj_val)
		max_pval = proj_val;
	    }
	  }
	}
	k = 0;
	for(int i = 0; i < pb_j.ColorBlock.rows; i++)
	{
	  for(int j = 0; j < pb_j.ColorBlock.cols; j++)
	  {
	    if(pb_j.BinaryBlock.data[i*pb_j.BinaryBlock.cols+j] != 255)
	    {
	      if(min_pval < 0.0)
	      {
		temp_pdata[k] = temp_pdata[k] - min_pval;
		max_pval = max_pval - min_pval;
	      }
	      k++;
	    }
	  }
	}	
	
	vector<Ray> tempR = SWT(pb_j.ColorBlock,pb_j.BinaryBlock);
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
	      
	      pb_j.valid = true;
	      
	      // Detecting Features for each valid blocks
	      
	      FeatureVec elem;
	      
	      //Vec10f elem;
	      
	      pb_j.SetSWSize(i);
	      
	      Mat tempstrokewidth = Mat(TSW.size(),1,CV_16UC1,TSW.data());
	      elem[0] =(float) FindMean(tempstrokewidth);
	      //BlockSWMean.push_back(FindMean(tempstrokewidth));
	      elem[1] =(float) FindStdDev(tempstrokewidth);
	      //BlockSWStdDev.push_back(FindStdDev(tempstrokewidth));
	      tempstrokewidth.release();
	      
	      int temp_pixel = NumberofForegroundPixel(pb_j.BinaryBlock);
	      Rect boundRect = pb_j.GetBoundRect();
	      
	      double tden = (temp_pixel/(boundRect.height*boundRect.width));
	    //  elem[4] =(float) tden; 
	      //density.push_back(tden);
	      tden = (temp_pixel)/(contourArea(pb_j.BlockContour));
	    //  elem[5] =(float) tden;
	      //density1.push_back(tden);
	      
	      //Height.push_back(boundRect[j].height);
	     // Width.push_back(boundRect[j].width);
	   //   elem[5] = boundRect.height *1.0;
	    //  elem[7] = boundRect.width * 1.0;
	      
	      //Blocks.push_back(TempC);
	      //TempC.release();
	      //Binary_blocks.push_back(TempB);
	      //TempB.release();
	      
	      Mat ProjDataBlock = Mat(temp_pdata.size(),1,CV_32FC1,temp_pdata.data());
	      temp_pdata.clear();
	      elem[2] =(float) FindMean(ProjDataBlock);
	     // BlockProjValMean.push_back(FindMean(ProjDataBlock));
	      //BlockProjValStdDev.push_back(FindStdDev(ProjDataBlock));
	      elem[3] =(float) FindStdDev(ProjDataBlock);
	      ProjDataBlock.release();
	      
	      Point2i tposi;
	      tposi.x = boundRect.x + (boundRect.width/2);
	      tposi.y = boundRect.y + (boundRect.height/2);
	    //  elem[6] = tposi.x * 1.0;
	    //  elem[7] = tposi.y * 1.0;
	      
	     // features.push_back(elem);
	      
	      pb_j.Setfeature(elem);
	    }
	    else
	      pb_j.valid = false;
	    
	    PB[m] = pb_j;
	    
	    
		
      }
      
      return(PB);
           
}


/*---------------------------------------------------------------AdjacencyMatrix----------------------------------------------------------------------------*/




/*@function FindAdjacencyMatrix
 * @param : input : vector<vector<type> > - A adjacency matrix with relation value between two node is given, in double
 * 		    double connection_condition - It is the value by checking which two nodes are connected
 * 
 * example : connection_condition = x;
 * 	     if(RelationMat.at<double>(i,j) > x)
 * 		
 * return : A AdjacencyMatrix of bool type (vector<vector<bool> >)
 * */

template <typename T>
vector<vector<bool> > FindAdjacencyMatrix(vector<vector<T> > &Relation, T connection_condition)
{
  vector<vector<bool> > AdjMat;
  for(int i=0;i<Relation.size();i++)
  {
    vector<bool> t;
    for(int j=0;j<Relation[i].size();j++)
    {
      
      if(Relation[i][j] > connection_condition)
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


template <typename T>
vector<vector<bool> > FindAdjacencyMatrix2data(vector<vector<T> > &Relation1, T connection_condition1, vector<vector<T> > &Relation2, T connection_condition2)
{
  vector<vector<bool> > AdjMat;
  for(int i=0;i<Relation1.size();i++)
  {
    vector<bool> t;
    for(int j=0;j<Relation1[i].size();j++)
    {
      if(Relation1[i][j] > connection_condition1 && Relation2[i][j] > connection_condition2)
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
template <typename T>
vector<T> FindIntersection(vector<T> &v1, vector<T> &v2)
{
  vector<T> v3;
  
  sort(v1.begin(), v1.end());
  sort(v2.begin(), v2.end());
  
  set_intersection(v1.begin(),v1.end(),v2.begin(),v2.end(),back_inserter(v3));
  
  return (v3);
}



/*-------------------------------------------------------------------------------------------------------------------------------------------*/

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

double twoSampleTtest_UnequalVariance(int sample1_size, int sample2_size, double sample1_mean, double sample2_mean, double sample1_sd, double sample2_sd)
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
  if(sigma>0)
    t = t/sigma;
    
  return(t);
}

double twoSampleTtest_EqualVariance(int sample1_size, int sample2_size, double sample1_mean, double sample2_mean, double sample1_sd, double sample2_sd)
{
  double sp;
  double sigma1,sigma2;
  sigma1 = sample1_sd * sample1_sd;
  sigma2 = sample2_sd * sample2_sd;
  sp = ((sample1_size - 1) * sigma1)+((sample2_size - 1) * sigma2);
  sp = sp/(sample1_size+sample2_size-2);
  double sigma;
  sigma = ((1/sample1_size)+(1/sample2_size));
  sigma = sp * sigma;
  sigma = sqrt(sigma);
  double t;
  t = sample1_mean - sample2_mean;
  if(sigma > 0)
    t = t/sigma;
  return(t);
}

int DegreeofFreedom_UnequalVariance(int sample1_size, int sample2_size, double sample1_sd, double sample2_sd)
{
  double neu1,neu2,neu;
  neu1 = (sample1_sd * sample1_sd)/sample1_size;
  neu2 = (sample2_sd * sample2_sd)/sample2_size;
  neu = neu1 + neu2;
  neu = neu * neu;
  double deno1;
  deno1 = (sample1_sd * sample1_sd)/sample1_size;
  deno1 = deno1 * deno1;
  deno1 = deno1/(sample1_size - 1);
  double deno2;
  deno2 = (sample2_sd * sample2_sd)/sample2_size;
  deno2 = deno2 * deno2;
  deno2 = deno2/(sample2_size - 1);
  
  double deno = deno1 + deno2;
  
  double val = neu/deno;
  val = val + 0.5;
  
  int dof =(int) floor(val);
  return(dof);
  
}

int DegreeofFreedom_EqualVariance(int sample1_size, int sample2_size)
{
  int dof = sample1_size + sample2_size - 2;
  return(dof);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------*/


clusinfo ClusteringSCCN(vector<ProcessingBlock> &PB, double ColorTh, double SWTh)
{
  vector<vector<double> > sw_testval;
  vector<vector<double> > color_testval;
  
  for(int i=0;i<PB.size();i++)
  {
    ProcessingBlock pb_i = PB[i];
    if(pb_i.valid)
    {
      FeatureVec elem_i = pb_i.Getfeature();
      vector<double> sw_tempval;
      vector<double> color_tempval;
      for(int j=0;j<PB.size();j++)
      {
	ProcessingBlock pb_j = PB[j];
	if(i!=j)
	{	  
	  if(pb_j.valid)
	  {
	    FeatureVec elem_j = pb_j.Getfeature();
	  //  printf("SW\tm1=%fm2=%fs1=%fs2=%f\n",elem_i[0],elem_j[0],elem_i[1],elem_j[1]); 
	    double tswval;
	    int sdof;
	    if(elem_i[1]==elem_j[1]) // In New vertion (F Test isapplied)
	    {
	      tswval = twoSampleTtest_EqualVariance(pb_i.GetSWSize(),pb_j.GetSWSize(),elem_i[0],elem_j[0],elem_i[1],elem_j[1]);
	      sdof = DegreeofFreedom_EqualVariance(pb_i.GetSWSize(),pb_j.GetSWSize());
	    }
	    else
	    {
	      tswval = twoSampleTtest_UnequalVariance(pb_i.GetSWSize(),pb_j.GetSWSize(),elem_i[0],elem_j[0],elem_i[1],elem_j[1]);
	      sdof = DegreeofFreedom_UnequalVariance(pb_i.GetSWSize(),pb_j.GetSWSize(),elem_i[1],elem_j[1]);
	    }
	    
	  //  printf("size 1 = %d\t size 2 = %d\t SW DOF = %d\t TVal = %lf\n",pb_i.GetSWSize(),pb_j.GetSWSize(),sdof,tswval);
	    tswval = studenttdistribution(sdof,tswval);
	    if(tswval > 0.5)
	      tswval =  2 * (1 - tswval);
	    else
	      tswval = 2 * tswval;
	    
	    sw_tempval.push_back(tswval);
	    
	    
	  //  printf("clr\tm1=%fm2=%fs1=%fs2=%f\n",elem_i[2],elem_j[3],elem_i[3],elem_j[3]);
	    double tclval;
	    int cdof;
	    if(elem_i[3]==elem_j[3])// In New vertion (F Test isapplied)
	    {
	      tclval = twoSampleTtest_EqualVariance(pb_i.GetColorBlkSize(),pb_j.GetColorBlkSize(),elem_i[2],elem_j[2],elem_i[3],elem_j[3]);
	      sdof = DegreeofFreedom_EqualVariance(pb_i.GetColorBlkSize(),pb_j.GetColorBlkSize());
	    }
	    else
	    {
	      tclval = twoSampleTtest_UnequalVariance(pb_i.GetColorBlkSize(),pb_j.GetColorBlkSize(),elem_i[2],elem_j[2],elem_i[3],elem_j[3]);
	      cdof = DegreeofFreedom_UnequalVariance(pb_i.GetColorBlkSize(),pb_j.GetColorBlkSize(),elem_i[3],elem_j[3]);
	    }
	    
	 //   printf("size 1 = %d\t size 2 = %d\t CL DOF = %d\t TVal = %lf\n",pb_i.GetColorBlkSize(),pb_j.GetColorBlkSize(),cdof,tclval);
	    tclval = studenttdistribution(cdof,tclval);
	    if(tclval > 0.5)
	      tclval =  2 * (1 - tclval);
	    else
	      tclval = 2 * tclval;
	    
	    color_tempval.push_back(tclval);	    
	    
	  }
	}
	else
	{
	  if(pb_j.valid)
	  {
	    sw_tempval.push_back(1.0);
	    color_tempval.push_back(1.0);
	  }
	}
      }
      sw_testval.push_back(sw_tempval);
      sw_tempval.clear();
      color_testval.push_back(color_tempval);
      color_tempval.clear();
    }
  }
  
  
  
  
  
  printf("Before Finding Adjacency Matrix of Projected data\n");
  
  
  
  vector<vector<bool> > ProjAdjMat;
 // ProjAdjMat = FindAdjacencyMatrix(PTtestVal, 0.05);
  ProjAdjMat = FindAdjacencyMatrix(color_testval, ColorTh);
  printf("After Finding Adjacency Matrix of Projected data\n");
  
  
  
  printf("Before Finding CC of Projected data\n");
  vector<vector<int> > Projcc;
  //vector<int> projlabels(number_of_valid_block);
  vector<int> projlabels(color_testval.size());
  
  
  DFSCC(ProjAdjMat, Projcc, projlabels);
  
  ProjAdjMat.clear();
  printf("After Finding CC of Projected data\n");
  
  
  printf("Number of ConnectedComponent with Projected Data is %d\n",Projcc.size());
  
  
  printf("Before Finding Adjacency Matrix of SW data\n");
  vector<vector<bool> > SWAdjMat;
  SWAdjMat = FindAdjacencyMatrix(sw_testval, SWTh);
  printf("After Finding Adjacency Matrix of SW data\n");
  
  vector<vector<int> > SWcc;
  vector<int> SWlabels(sw_testval.size());
  
  printf("Before Finding CC of SW data\n");
  
  DFSCC(SWAdjMat, SWcc, SWlabels);
  printf("After Finding CC of SW data\n");
  SWAdjMat.clear();
  
  printf("Number of ConnectedComponent with SW Data is %d\n",SWcc.size());
  
  
   vector<vector<int> > SCCN_CC;
   vector<int> SCCN_labels(color_testval.size());
   int k = 0;
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
	   SCCN_labels[t3[m]] = k;
	 SCCN_CC.push_back(t3);
	 k = k + 1;
       }
       t3.clear();
       
       t2.clear();
     }
     t1.clear();
   }
   
   
   clusinfo SCCN;
   SCCN.clusters = SCCN_CC;
   SCCN_CC.clear();
   SCCN.clusternum = SCCN_labels;
   SCCN_labels.clear();
   
   return(SCCN);
  
  
}


clusinfo ClusteringSCE(vector<ProcessingBlock> &PB, double ColorTh, double SWTh)
{
  vector<vector<double> > sw_testval;
  vector<vector<double> > color_testval;
  
  for(int i=0;i<PB.size();i++)
  {
    ProcessingBlock pb_i = PB[i];
    if(pb_i.valid)
    {
      FeatureVec elem_i = pb_i.Getfeature();
      vector<double> sw_tempval;
      vector<double> color_tempval;
      for(int j=0;j<PB.size();j++)
      {
	ProcessingBlock pb_j = PB[j];
	if(i!=j)
	{
	    if(pb_j.valid)
	    {
	      FeatureVec elem_j = pb_j.Getfeature();
	      //printf("SW\tm1=%f m2=%f s1=%f s2=%f\n",elem_i[0],elem_j[0],elem_i[1],elem_j[1]); 
	      double tswval;
	      int sdof;
	      if(elem_i[1]==elem_j[1]) // In New vertion (F Test isapplied)
	      {
		tswval = twoSampleTtest_EqualVariance(pb_i.GetSWSize(),pb_j.GetSWSize(),elem_i[0],elem_j[0],elem_i[1],elem_j[1]);
		sdof = DegreeofFreedom_EqualVariance(pb_i.GetSWSize(),pb_j.GetSWSize());
	      }
	      else
	      {
		tswval = twoSampleTtest_UnequalVariance(pb_i.GetSWSize(),pb_j.GetSWSize(),elem_i[0],elem_j[0],elem_i[1],elem_j[1]);
		sdof = DegreeofFreedom_UnequalVariance(pb_i.GetSWSize(),pb_j.GetSWSize(),elem_i[1],elem_j[1]);
	      }
	      
	      //printf("size 1 = %d\t size 2 = %d\t SW DOF = %d\t TVal = %lf\n",pb_i.GetSWSize(),pb_j.GetSWSize(),sdof,tswval);
	      tswval = studenttdistribution(sdof,tswval);
	      if(tswval > 0.5)
		tswval =  2 * (1 - tswval);
	      else
		tswval = 2 * tswval;
	      
	      sw_tempval.push_back(tswval);
	      
	      
	     // printf("clr\tm1=%f m2=%f s1=%f s2=%f\n",elem_i[2],elem_j[2],elem_i[3],elem_j[3]);
	      double tclval;
	      int cdof;
	      if(elem_i[3]==elem_j[3]) // In New vertion (F Test isapplied)
	      {
		tclval = twoSampleTtest_EqualVariance(pb_i.GetColorBlkSize(),pb_j.GetColorBlkSize(),elem_i[2],elem_j[2],elem_i[3],elem_j[3]);
		sdof = DegreeofFreedom_EqualVariance(pb_i.GetColorBlkSize(),pb_j.GetColorBlkSize());
	      }
	      else
	      {
		tclval = twoSampleTtest_UnequalVariance(pb_i.GetColorBlkSize(),pb_j.GetColorBlkSize(),elem_i[2],elem_j[2],elem_i[3],elem_j[3]);
		cdof = DegreeofFreedom_UnequalVariance(pb_i.GetColorBlkSize(),pb_j.GetColorBlkSize(),elem_i[3],elem_j[3]);
	      }
	      
	     // printf("size 1 = %d\t size 2 = %d\t CL DOF = %d\t TVal = %lf\n",pb_i.GetColorBlkSize(),pb_j.GetColorBlkSize(),cdof,tclval);
	      tclval = studenttdistribution(cdof,tclval);
	      if(tclval > 0.5)
		tclval =  2 * (1 - tclval);
	      else
		tclval = 2 * tclval;
	      
	      color_tempval.push_back(tclval);	    
	    }
	}
	else
	{
	  if(pb_j.valid)
	  {
	    sw_tempval.push_back(1.0);
	    color_tempval.push_back(1.0);
	  }
	}
      }
      sw_testval.push_back(sw_tempval);
      sw_tempval.clear();
      color_testval.push_back(color_tempval);
      color_tempval.clear();
    }
  }
  
  
  
  vector<vector<bool> > AdjMat;

  AdjMat = FindAdjacencyMatrix2data(color_testval, ColorTh, sw_testval, SWTh);
  vector<vector<int> > SCE_CC;
  vector<int> SCE_labels(color_testval.size());
  DFSCC(AdjMat, SCE_CC, SCE_labels);
  
  
  clusinfo SCE;
   SCE.clusters = SCE_CC;
   SCE_CC.clear();
   SCE.clusternum = SCE_labels;
   SCE_labels.clear();
   
   return(SCE);
  
  
}



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
      binarization_type = 5;
      
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
	 // printf("Max value of plane %d is %lf\n",i,temp_max_val);
	}

      
      
      vector<float> colfeature = ScalarColorFeatureMasked(hsv_planes, Gt_binary_dst);
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
	    proj_val = (colfeature[0] * hsv_planes[0].at<uchar>(i,j) ) + ( colfeature[1] * hsv_planes[1].at<uchar>(i,j) ) + ( colfeature[2] * hsv_planes[2].at<uchar>(i,j) );
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
		//elem[5] = boundRect[j].height *1.0;
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
		//elem[6] = tposi.x * 1.0;
		//elem[7] = tposi.y * 1.0;
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



/**
 * @func GetConfusionMatrix
 * @param vector<ProcessingBlock> : Send ProcessingBlock after Prediction
 * @param TDC : Send the Training Data Class
 * @return vector<ConfusionMatrix> : Return Confution matrix for the Processing blocks
 * */
vector<ConfusionMatrix> GetConfusionMatrix(vector<ProcessingBlock> PB, vector<int> ClassNumber)
{
      vector<ConfusionMatrix> CM(ClassNumber.size());
      
      for(int i=0;i<CM.size();i++)
      {
	CM[i].initialize();  
	CM[i].initializeMulticlass(ClassNumber.size());
      }
      
      for(int i=0;i<PB.size();i++)
      {
	ProcessingBlock pb_i = PB[i];
	if(pb_i.valid)
	{
	  int GtClass = pb_i.GetClassLabel();
	  int response = pb_i.GetPredictedLabel();
	
	  CM[response].multiclassCM[GtClass] = CM[response].multiclassCM[GtClass] + 1;
	  
	  for(int j=0;j<ClassNumber.size();j++)
	  {
	    ConfusionMatrix cm_j = CM[j];
	    
	    if(response == ClassNumber[j]) // response == A Class j (Positive part (tp,fp))
	    {
	      if(ClassNumber[j] == GtClass) // GT == class j => tp
	      {
		cm_j.tp = cm_j.tp + 1;
		CM[j] = cm_j;
	      }
	      else // GT != class j => fp
	      {
		
		cm_j.fp = cm_j.fp + 1;
		CM[j] = cm_j;
		//printf("Block %d fp = %d of %d\n",i,cm_knn_j.fp,j);
	      }
	    }
	    else // response != Class j (Negetive part (tn,fn))
	    { 
	      if(ClassNumber[j] == GtClass) // GT == class j ==> fn
	      {
		
		cm_j.fn = cm_j.fn + 1;
		CM[j] = cm_j;
	      // printf("Block %d fn = %d of %d\n",i,cm_knn_j.fn,j);
	      }
	      else // GT != class =>  tn
	      {
		cm_j.tn = cm_j.tn + 1;
		CM[j] = cm_j;
	      }
	    }

	  
	  }
	}
      }
      
      
      return(CM);
}


#if _KNN_

void segmentation_withKNN(vector<ProcessingBlock> &PB, TDC Data, int K)
{
   
     char *foldername;
    
    foldername = "KNN";
  
    makedir(foldername);
    
    FILE *labelname;
    char * tempname;
    
     tempname = (char *) malloc ( 2001 * sizeof(char));
      if(tempname == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      strcpy(tempname,substring);
      strcat(tempname,"_labels.csv");
      
      char *name;
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(foldername,tempname);
         
      labelname = fopen(name,"w");  
  
  
      printf("In KNN with K = %d\n",K);
    #if defined HAVE_OPENCV_OCL && _OCL_KNN_
      cv::ocl::KNearestNeighbour knnClassifier;
      Mat temp, result;
      knnClassifier.train(Data.TrainData, Data.TrainClass, temp, false, K);
      cv::ocl::oclMat testSample_ocl, reslut_ocl;
    #else
      CvKNearest knnClassifier( Data.TrainData, Data.TrainClass, Mat(), false, K );
    #endif
      
      
      for(int i=0;i<PB.size();i++)
      {
	ProcessingBlock pb_i = PB[i];
	if(pb_i.valid)
	{
	  FeatureVec elem = pb_i.Getfeature();
	  
	  
	  
	  Mat TestData;
	  Mat(elem).copyTo(TestData);
	  //printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
	  transpose(TestData,TestData);
	  
	  int response;
	  
	  #if defined HAVE_OPENCV_OCL && _OCL_KNN_
	  
	    testSample_ocl.upload(TestData);
	    knnClassifier.find_nearest(testSample_ocl, K, reslut_ocl);

	    reslut_ocl.download(result);
	    response = saturate_cast<int>(result.at<float>(0));
	  
	  #else
	    
	     response = (int)knnClassifier.find_nearest( TestData, K );
	    
	  #endif
	  
	  
	PB[i].SetPredictedLabel(response);

	}
      }
    
      
 
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetClassLabel());
      }     
    }
    fprintf(labelname," \n");
    
    
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetPredictedLabel());
      }
    }
    fprintf(labelname," \n");
    
    fclose(labelname);
    
      
}

#endif


#if _NBC_

void segmentation_withNBC(vector<ProcessingBlock> &PB, TDC Data)
{
  
      char *foldername;
    
    foldername = "NBC";
  
    makedir(foldername);
    
    FILE *labelname;
    char * tempname;
    
     tempname = (char *) malloc ( 2001 * sizeof(char));
      if(tempname == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      strcpy(tempname,substring);
      strcat(tempname,"_labels.csv");
      
      char *name;
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(foldername,tempname);
         
      labelname = fopen(name,"w");
  
  
  CvNormalBayesClassifier normalBayesClassifier( Data.TrainData, Data.TrainClass );

    
    printf("In NBC \n");
    //vector<int> ResposeNBC;
    
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	FeatureVec elem = pb_i.Getfeature();
	Mat TestData;
	Mat(elem).copyTo(TestData);
	//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
	transpose(TestData,TestData);
	//TestData = TestData.reshape( 1, TestData.rows );
	//TestData.convertTo( TestData, CV_32FC1 );
	
	//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
	//printf("Gt Class = %d\n",pb_i.GetClassLabel());
	
	int response = (int)normalBayesClassifier.predict( TestData );
	
	//printf("Gt Class = %d\tPredicted Class = %d\n",pb_i.GetClassLabel(),response);
	
	PB[i].SetPredictedLabel(response);
	

      }
    }
    
    
 
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetClassLabel());
      }     
    }
    fprintf(labelname," \n");
    
    
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetPredictedLabel());
      }
    }
    fprintf(labelname," \n");
    
    fclose(labelname);
    
}

#endif

#if _EM_
  
void segmentation_withEM(vector<ProcessingBlock> &PB, TDC Data)
{
  
         char *foldername;
    
    foldername = "EM";
  
    makedir(foldername);
    
    FILE *labelname;
    char * tempname;
    
     tempname = (char *) malloc ( 2001 * sizeof(char));
      if(tempname == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      strcpy(tempname,substring);
      strcat(tempname,"_labels.csv");
      
      char *name;
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(foldername,tempname);
         
      labelname = fopen(name,"w");    

    //  printf("in EM \n");
    
    printf("In Em \n");
  
      vector<cv::EM> em_models(Data.ClassNumber.size());
      
      Mat trainSamples, trainClasses;
      trainSamples = Data.TrainData;
      trainClasses = Data.TrainClass;
      
      CV_Assert((int)trainClasses.total() == trainSamples.rows);
      CV_Assert((int)trainClasses.type() == CV_32SC1);
      
     // vector<int> ResponseEM;

      for(size_t modelIndex = 0; modelIndex < em_models.size(); modelIndex++)
      {
	//printf("Em Training %d\n",modelIndex);
	  const int componentCount = 3;
	  em_models[modelIndex] = EM(componentCount, cv::EM::COV_MAT_DIAGONAL);

	  Mat modelSamples;
	  for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
	  {
	      if(trainClasses.at<int>(sampleIndex) == (int)modelIndex)
		  modelSamples.push_back(trainSamples.row(sampleIndex));
	  }

	  // learn models
	  if(!modelSamples.empty())
	      em_models[modelIndex].train(modelSamples);
      }
	  
	  
    
	  for(int i=0;i<PB.size();i++)
	  {
	    ProcessingBlock pb_i = PB[i];
	     if(pb_i.valid)
	     {
		FeatureVec elem = pb_i.Getfeature();
		Mat TestData;
		Mat(elem).copyTo(TestData);
		//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
		transpose(TestData,TestData);
		//TestData = TestData.reshape( 1, TestData.rows );
		//TestData.convertTo( TestData, CV_32FC1 );
	
		//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
		
		Mat logLikelihoods(1, em_models.size(), CV_64FC1, Scalar(-DBL_MAX));
		for(size_t modelIndex = 0; modelIndex < em_models.size(); modelIndex++)
		{
		    if(em_models[modelIndex].isTrained())
			logLikelihoods.at<double>(modelIndex) = em_models[modelIndex].predict(TestData)[0];
		}
		Point maxLoc;
		minMaxLoc(logLikelihoods, 0, 0, 0, &maxLoc);

		int response = maxLoc.x;
		PB[i].SetPredictedLabel(response);

	     }
	    
	  }
	
	
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetClassLabel());
      }     
    }
    fprintf(labelname," \n");
    
    
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetPredictedLabel());
      }
    }
    fprintf(labelname," \n");
    
    fclose(labelname);
	  
   
}
  
#endif  

#if _SVM_

void segmentation_withSVM(vector<ProcessingBlock> &PB, TDC Data)
{  
    
    CvSVMParams params;
            params.svm_type = CvSVM::C_SVC;
            params.kernel_type = CvSVM::POLY; //CvSVM::LINEAR;
            params.degree = 0.5;
            params.gamma = 1;
            params.coef0 = 1;
            params.C = 6;
            params.nu = 0.5;
            params.p = 0;
            params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
  
    char *foldername;
    
    foldername = "SVM";
  
    makedir(foldername);
    
    FILE *labelname;
    char * tempname;
    
     tempname = (char *) malloc ( 2001 * sizeof(char));
      if(tempname == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      strcpy(tempname,substring);
      strcat(tempname,"_labels.csv");
      
      char *name;
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(foldername,tempname);
         
      labelname = fopen(name,"w");  
  
  
    
       // learn classifier
#if defined HAVE_OPENCV_OCL && _OCL_SVM_
    cv::ocl::CvSVM_OCL svmClassifier(trainSamples, trainClasses, Mat(), Mat(), params);
#else
    CvSVM svmClassifier( Data.TrainData, Data.TrainClass, Mat(), Mat(), params );
#endif
    
    
    printf("In SVM \n");
    
    
    
    
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	FeatureVec elem = pb_i.Getfeature();
	Mat TestData;
	Mat(elem).copyTo(TestData);
	//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
	transpose(TestData,TestData);
	//TestData = TestData.reshape( 1, TestData.rows );
	//TestData.convertTo( TestData, CV_32FC1 );
	
	//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
	
	int response = (int)svmClassifier.predict( TestData );
	PB[i].SetPredictedLabel(response);
	
      }
    }
    
    
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetClassLabel());
      }     
    }
    fprintf(labelname," \n");
    
    
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetPredictedLabel());
      }
    }
    fprintf(labelname," \n");
    
    fclose(labelname);

    
}

#endif


    


#if _DT_


void segmentation_withDT(vector<ProcessingBlock> &PB, TDC Data)
{ 
  
    char *foldername;
    
    foldername = "DT";
  
    makedir(foldername);
    
    FILE *labelname;
    char * tempname;
    
     tempname = (char *) malloc ( 2001 * sizeof(char));
      if(tempname == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      strcpy(tempname,substring);
      strcat(tempname,"_labels.csv");
      
      char *name;
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(foldername,tempname);
         
      labelname = fopen(name,"w");
    
    
    CvDTree  dtree;

    Mat var_types( 1, Data.TrainData.cols + 1, CV_8UC1, Scalar(CV_VAR_ORDERED) );
    var_types.at<uchar>( Data.TrainData.cols ) = CV_VAR_CATEGORICAL;

    CvDTreeParams params;
    params.max_depth = 8;
    params.min_sample_count = 2;
    params.use_surrogates = false;
    params.cv_folds = 0; // the number of cross-validation folds
    params.use_1se_rule = false;
    params.truncate_pruned_tree = false;

    dtree.train( Data.TrainData, CV_ROW_SAMPLE, Data.TrainClass,
                 Mat(), Mat(), var_types, Mat(), params );
    
    
     printf("In DT \n");
    
    
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	FeatureVec elem = pb_i.Getfeature();
	Mat TestData;
	Mat(elem).copyTo(TestData);
	//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
	transpose(TestData,TestData);
	//TestData = TestData.reshape( 1, TestData.rows );
	//TestData.convertTo( TestData, CV_32FC1 );
	
	//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
	
	int response = (int)dtree.predict( TestData )->value;
	PB[i].SetPredictedLabel(response);

      }
    }

    
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetClassLabel());
      }     
    }
    fprintf(labelname," \n");
    
    
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetPredictedLabel());
      }
    }
    fprintf(labelname," \n");
    
    fclose(labelname);

}

#endif
    
    
#if _RF_
    
void segmentation_withRF(vector<ProcessingBlock> &PB, TDC Data)
{
    
    char *foldername;
    
    foldername = "RF";
  
    makedir(foldername);
    
    FILE *labelname;
    char * tempname;
    
     tempname = (char *) malloc ( 2001 * sizeof(char));
      if(tempname == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      strcpy(tempname,substring);
      strcat(tempname,"_labels.csv");
      
      char *name;
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(foldername,tempname);
         
      labelname = fopen(name,"w");
    
   
    
    CvRTrees  rtrees;
    CvRTParams  params( 8, // max_depth,
                        2, // min_sample_count,
                        0.f, // regression_accuracy,
                        false, // use_surrogates,
                        16, // max_categories,
                        0, // priors,
                        false, // calc_var_importance,
                        1, // nactive_vars,
                        5, // max_num_of_trees_in_the_forest,
                        0, // forest_accuracy,
                        CV_TERMCRIT_ITER // termcrit_type
                       );

    rtrees.train( Data.TrainData, CV_ROW_SAMPLE, Data.TrainClass, Mat(), Mat(), Mat(), Mat(), params );
    
    
     printf("In RF \n");
    
     for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	FeatureVec elem = pb_i.Getfeature();
	Mat TestData;
	Mat(elem).copyTo(TestData);
	//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
	transpose(TestData,TestData);
	//TestData = TestData.reshape( 1, TestData.rows );
	//TestData.convertTo( TestData, CV_32FC1 );
	
	//printf("Rows = %d Cols = %d channel = %d\n",TestData.rows,TestData.cols,TestData.channels());
	
	int response = (int)rtrees.predict( TestData );
	PB[i].SetPredictedLabel(response);
	
	
      }
    }

     
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetClassLabel());
      }     
    }
    fprintf(labelname," \n");
    
    
    for(int i=0;i<PB.size();i++)
    {
      ProcessingBlock pb_i = PB[i];
      if(pb_i.valid)
      {
	fprintf(labelname,"%d,",pb_i.GetPredictedLabel());
      }
    }
    fprintf(labelname," \n");
    
    fclose(labelname);

}

#endif


void classification(vector<ProcessingBlock> PB, TDC Data)
{
  printf("In segmentation \n");
  
  
  char *name,*output,*tempname;
  
  
      char *groundtruthdest;
      groundtruthdest = "300dpi/groundtruth";
      
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
      
      
      for(int i=0;i<PB.size();i++)
      {
	ProcessingBlock pb_i = PB[i];
	if(pb_i.valid)
	{
	  int gtlabel;
	  vector<Point> contours_poly = pb_i.GetApproxPoly();
	  Rect boundRect = pb_i.GetBoundRect();
	  for(int m=boundRect.y;m<boundRect.y+boundRect.height;m++)
	  {
	    int n;
	    bool brk = false;
	    for(n=boundRect.x;n<boundRect.x+boundRect.width;n++)
	    {
	      bool measure_dist;
	      if((pointPolygonTest(contours_poly,Point(n,m),measure_dist) > 0.0) && groundtruthdata.data[m*groundtruthdata.cols+n]!=255)
	      {
		gtlabel = groundtruthdata.data[m*groundtruthdata.cols+n];
		brk = true;
		//printf("Label = %d\n",gtlabel);
		break;
	      }
	    }
	    if(brk)
	      break;
	  }
	  pb_i.SetClassLabel(gtlabel);
	  PB[i] = pb_i;
	  FeatureVec elem = pb_i.Getfeature();
	  //fprintf(sw_feature,"%f,%f,%f,%f,%s\n",elem[0],elem[1],elem[2],elem[3], GetClassName(pb_i.GetClassLabel()));
	  //fprintf(test_features,"%f,%f,%f,%f,%s\n",elem[0],elem[1],elem[2],elem[3], GetClassName(pb_i.GetClassLabel()));
	}
      }
      
      
  ConfusionMatrix overall;
  
  
  #if _NBC_
    
    segmentation_withNBC(PB,Data);
    
    vector<ConfusionMatrix> CM_NBC = GetConfusionMatrix(PB,Data.ClassNumber);

    
    printf("NBC Done\n");
    
    overall.initialize();
    
    fprintf(f_nbc,"%s\t",substring);
    printf("%s\n",substring);
    for(int i=0;i<CM_NBC.size();i++)
    {
      ConfusionMatrix cm_i = CM_NBC[i];
      fprintf(f_nbc,"%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
    //  PrintClassName(Data.ClassNumber[i]);
      printf("tp = %d\t fp = %d\t tn = %d\t fn = %d\t precesion = %f\t recall = %f\t accuracy = %f\n",cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
      overall.tp = overall.tp + cm_i.tp;
      overall.fp = overall.fp + cm_i.fp;
      overall.tn = overall.tn + cm_i.tn;
      overall.fn = overall.fn + cm_i.fn;

    }
    fprintf(f_nbc,"overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
  //  printf("overall\n tp = %d\t fp = %d\t tn = %d\t fn = %d\t precession = %f\trecall = %f\taccuracy = %f\n",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
    fprintf(f_nbc,"\n");
 
  #endif
  
    
    
  #if _EM_
  
   segmentation_withEM(PB,Data);
   
   vector<ConfusionMatrix> CM_EM = GetConfusionMatrix(PB,Data.ClassNumber);
   
   printf("EM Done\n");
   
   overall.initialize();
   
   fprintf(f_em,"%s\t",substring);
   printf("%s\n",substring);
    for(int i=0;i<CM_EM.size();i++)
    {
      ConfusionMatrix cm_i = CM_EM[i];
      fprintf(f_em,"%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
     // PrintClassName(Data.ClassNumber[i]);
     // printf("tp = %d\t fp = %d\t tn = %d\t fn = %d\t precesion = %f\t recall = %f\t accuracy = %f\n");
      overall.tp = overall.tp + cm_i.tp;
      overall.fp = overall.fp + cm_i.fp;
      overall.tn = overall.tn + cm_i.tn;
      overall.fn = overall.fn + cm_i.fn;

    }
    fprintf(f_em,"overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
   // printf("overall\n tp = %d\t fp = %d\t tn = %d\t fn = %d\t precession = %f\trecall = %f\taccuracy = %f\n",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
    fprintf(f_em,"\n");
  
  #endif
   
   
#if _KNN_
   

   segmentation_withKNN(PB,Data,K);
   
   vector<ConfusionMatrix> CM_KNN = GetConfusionMatrix(PB,Data.ClassNumber);
   
   printf("KNN Done\n");
   
   overall.initialize();
   
   fprintf(f_knn,"%s\t",substring);
    for(int i=0;i<CM_KNN.size();i++)
    {
      ConfusionMatrix cm_i = CM_KNN[i];
      fprintf(f_knn,"%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
     // printf("%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
      overall.tp = overall.tp + cm_i.tp;
      overall.fp = overall.fp + cm_i.fp;
      overall.tn = overall.tn + cm_i.tn;
      overall.fn = overall.fn + cm_i.fn;

    }
    fprintf(f_knn,"overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
    //printf("overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
    fprintf(f_knn,"\n");
   
#endif
    
#if _SVM_
    
    
	    
	    segmentation_withSVM(PB,Data);
	    
	    vector<ConfusionMatrix> CM_SVM = GetConfusionMatrix(PB,Data.ClassNumber);
	    
	    printf("SVM Done\n");
	    
	    overall.initialize();
	    
	    
	    fprintf(f_svm,"%s\t",substring);
	    for(int i=0;i<CM_SVM.size();i++)
	    {
	      ConfusionMatrix cm_i = CM_SVM[i];
	      fprintf(f_svm,"%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
	     // printf("%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
	      overall.tp = overall.tp + cm_i.tp;
	      overall.fp = overall.fp + cm_i.fp;
	      overall.tn = overall.tn + cm_i.tn;
	      overall.fn = overall.fn + cm_i.fn;

	    }
	    fprintf(f_svm,"overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
	    //printf("overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
	    fprintf(f_svm,"\n");
	    
    
#endif
 
#if _DT_
  
	    
  segmentation_withDT(PB,Data);
  
  vector<ConfusionMatrix> CM_DT = GetConfusionMatrix(PB,Data.ClassNumber);
  
  printf("DT Done\n");
  
  fprintf(f_dt,"%s\t",substring);
  
  overall.initialize();
  
    for(int i=0;i<CM_DT.size();i++)
    {
      ConfusionMatrix cm_i = CM_DT[i];
      fprintf(f_dt,"%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
      //printf("%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
      overall.tp = overall.tp + cm_i.tp;
      overall.fp = overall.fp + cm_i.fp;
      overall.tn = overall.tn + cm_i.tn;
      overall.fn = overall.fn + cm_i.fn;

    }
    fprintf(f_dt,"overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
    //printf("overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
    fprintf(f_dt,"\n");
#endif
  
#if _RF_
    
  segmentation_withRF(PB,Data);
  
  vector<ConfusionMatrix> CM_RF = GetConfusionMatrix(PB,Data.ClassNumber);
  
  printf("RF Done\n");
  
  fprintf(f_rf,"%s\t",substring);
  
  overall.initialize();
  
    for(int i=0;i<CM_RF.size();i++)
    {
      ConfusionMatrix cm_i = CM_RF[i];
      fprintf(f_rf,"%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
     // printf("%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",Data.ClassNumber[i],cm_i.tp,cm_i.fp,cm_i.tn,cm_i.fn,cm_i.GetPrecesion(),cm_i.GetRecall(),cm_i.GetAccuracy());
      overall.tp = overall.tp + cm_i.tp;
      overall.fp = overall.fp + cm_i.fp;
      overall.tn = overall.tn + cm_i.tn;
      overall.fn = overall.fn + cm_i.fn;

    }
    fprintf(f_rf,"overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
    //printf("overall\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t",overall.tp,overall.fp,overall.tn,overall.fn,overall.GetPrecesion(),overall.GetRecall(),overall.GetAccuracy());
    fprintf(f_rf,"\n");
#endif
  
  
}


template <typename T>
int FindPosition(vector<T> x, T val)
{
  for(int i=0;i<x.size();i++)
  {
    if(x[i]==val)
    {
      return(i);
      break;
    }
  }
  printf("There is Some problem\n");
  exit(0);
}

void PBLabelsWRTClusterLabels(clusinfo Algo, vector<ProcessingBlock> &PB, vector<int> ClassNumber)
{
  int blk = 0;
  vector<int> preditedlabels;
  
  blk = 0;
  for(int i=0;i<PB.size();i++)
  {
    ProcessingBlock pb_i = PB[i];
    if(pb_i.valid)
    {
      preditedlabels.push_back(pb_i.GetPredictedLabel());
      blk = blk + 1;
    }  
  }
  
  vector<int> temphist(ClassNumber.size(),0);
  vector<vector<int> > ClusterClassHist(Algo.clusters.size(),temphist);
  temphist.clear();
  
  vector<int> newlabels(preditedlabels.size());
  
  for(int i=0;i<Algo.clusters.size();i++)
  {
    vector<int> cluster_i = Algo.clusters[i];
    for(int j=0;j<cluster_i.size();j++)
    {
      int label = preditedlabels[cluster_i[j]];
      int posi = FindPosition<int>(ClassNumber,label);
      
      ClusterClassHist[i][label] = ClusterClassHist[i][label] + 1;
    }
    int max = ClusterClassHist[i][0];
    int clusterlabel = 0;
    for(int j=0;j<ClusterClassHist[i].size();j++)
    {
      if(max < ClusterClassHist[i][j])
      {
	max = ClusterClassHist[i][j];
	clusterlabel = j;
      }
    }
    for(int j=0;j<cluster_i.size();j++)
    {
      newlabels[cluster_i[j]] = clusterlabel;
    }
  }
  
  blk = 0;
  for(int i=0;i<PB.size();i++)
  {
    ProcessingBlock pb_i = PB[i];
    if(pb_i.valid)
    {
      pb_i.SetPredictedLabel(newlabels[blk]);
      blk = blk + 1;
      PB[i] = pb_i;
    }
  }
  
  
}


void Clustering_Classification(clusinfo Algo, vector<ProcessingBlock> &PB, TDC Data, char *ClassifierName)
{
  if(ClassifierName == "NBC")
  {
    segmentation_withNBC(PB,Data);
    PBLabelsWRTClusterLabels(Algo, PB, Data.ClassNumber);
  }
  
  if(ClassifierName == "KNN")
  {
    K = 10;
    segmentation_withKNN(PB,Data,K);
    PBLabelsWRTClusterLabels(Algo, PB, Data.ClassNumber);
  }
  
  if(ClassifierName == "EM")
  {
    segmentation_withEM(PB,Data);
    PBLabelsWRTClusterLabels(Algo, PB, Data.ClassNumber);
  }
  
  if(ClassifierName == "DT")
  {
    segmentation_withDT(PB,Data);
    PBLabelsWRTClusterLabels(Algo, PB, Data.ClassNumber);
  }
  
  if(ClassifierName == "RF")
  {
    segmentation_withRF(PB,Data);
    PBLabelsWRTClusterLabels(Algo, PB, Data.ClassNumber);
  }
  
  if(ClassifierName == "SVM")
  {
    segmentation_withSVM(PB,Data);
    PBLabelsWRTClusterLabels(Algo, PB, Data.ClassNumber);
  }
  
}


/*-------------------------------------------------------------------------------------------------------------------------------------------*/

/**
 * @function readme
 */
void readmeknn()
{ std::cout << " Usage: ./clustering <train image-name- in text file> <Test image name in text file> <K for KNN>" << std::endl; }

void readmeother()
{ std::cout << " Usage: ./clustering <train image-name- in text file> <Test image name in text file> <color confidence> <strokewidth confidence> <clusteringName(SCCN/SCE)>" << std::endl; }



/*-------------------------------------------------MAIN--------------------------------------------------------------------------------------*/




int main(int argc, char *argv[])
{
  
  if(argc!=6)
  {readmeother(); return -1; }
  
  char *parentfolder;
  
  if(strcmp(argv[5],"SCCN") == 0)
  {
    SCCN_flag =true;
    parentfolder = (char *)malloc(2000*sizeof(char));
    parentfolder = "SCCN";
  }
  if(strcmp(argv[5],"SCE") == 0)
  {
    SCE_flag =true;
    parentfolder = (char *)malloc(2000*sizeof(char));
    parentfolder = "SCE";
  }
  
  makedir(parentfolder);
  
  
  
  int cth = atoi(argv[3]);
  int sth = atoi(argv[4]);
  
  double ColorTh =(double) (cth*1.0)/100;
  double SWTh = (double) (sth*1.0)/100;
  
  
  TDC TrainingData = Training(argv[1]);
  
 vector<char*> ClassifierName;
 ClassifierName.push_back("NBC");
 ClassifierName.push_back("KNN");
 ClassifierName.push_back("EM");
 ClassifierName.push_back("SVM");
 ClassifierName.push_back("DT");
 ClassifierName.push_back("RF");
 
 char *ClassifierFolder;
 
 for(int i=0;i<ClassifierName.size();i++)
 {
   printf("Classifier Name %s\n",ClassifierName[i]);
   ClassifierFolder = (char *)malloc(2000*sizeof(char));
   ClassifierFolder = CreateNameIntoFolder(parentfolder,ClassifierName[i]);
   makedir(ClassifierFolder);
 }
 
 
 ConfusionMatrix C;
 C.initialize();
 vector<ConfusionMatrix> CM_ALL(TrainingData.ClassNumber.size(),C);
 vector<vector<ConfusionMatrix> > CM_Classifier(ClassifierName.size(),CM_ALL);
 CM_ALL.clear();
 
  
  FILE *f;
  f = fopen(argv[2],"r");
 while(!feof(f))
 {
    //filename = (char *)malloc(2000*sizeof(char));
    char filename[2000];
    fscanf(f,"%s",&filename);
    printf("%s\n",filename);
  
  
  
   
    
    char *inputdst;
    // inputdst = "600dpi/scans";
      inputdst = "300dpi/scans";
      
      
      // substring = input_image_name_cut(argv[1]);
      substring = input_image_name_cut(filename);
      //makedir(substring);
      
    char *name,*tempname;
      
      name = (char *) malloc ( 2001 * sizeof(char));
      if(name == NULL)
      {
	printf("Memory can not be allocated\n");
	exit(0);
      }
      
      name = CreateNameIntoFolder(inputdst,filename);
      
      Mat src = imread(name,1);
      
     vector<ProcessingBlock> PB = GetProcessingBlocksWithFeatures(src);
     
     
     
  
  
      char *groundtruthdest;
      groundtruthdest = "300dpi/groundtruth";
      
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
      
      
      for(int i=0;i<PB.size();i++)
      {
	ProcessingBlock pb_i = PB[i];
	if(pb_i.valid)
	{
	  int gtlabel;
	  vector<Point> contours_poly = pb_i.GetApproxPoly();
	  Rect boundRect = pb_i.GetBoundRect();
	  for(int m=boundRect.y;m<boundRect.y+boundRect.height;m++)
	  {
	    int n;
	    bool brk = false;
	    for(n=boundRect.x;n<boundRect.x+boundRect.width;n++)
	    {
	      bool measure_dist;
	      if((pointPolygonTest(contours_poly,Point(n,m),measure_dist) > 0.0) && groundtruthdata.data[m*groundtruthdata.cols+n]!=255)
	      {
		gtlabel = groundtruthdata.data[m*groundtruthdata.cols+n];
		brk = true;
		//printf("Label = %d\n",gtlabel);
		break;
	      }
	    }
	    if(brk)
	      break;
	  }
	  pb_i.SetClassLabel(gtlabel);
	  PB[i] = pb_i;
	}
      }
     
      
      
     if(SCCN_flag)
     {
       clusinfo SCCN = ClusteringSCCN(PB,ColorTh,SWTh);
       segmentation_withSVM(PB,TrainingData);
       int tmp = 0;
       for(int i=0;i<PB.size();i++)
       {
	 ProcessingBlock pb_i = PB[i];
	 if(pb_i.valid)
	 {
	   printf("Gt = %d Cluster = %d SVM = %d\n",pb_i.GetClassLabel(),SCCN.clusternum[tmp],pb_i.GetPredictedLabel());
	   tmp = tmp + 1;
	 }
       }
       exit(0);
       
       for(int i=0;i<ClassifierName.size();i++)
       {
	 
	 ClassifierFolder = (char *)malloc(2000*sizeof(char));
	 
	 ClassifierFolder = CreateNameIntoFolder(parentfolder,ClassifierName[i]);
	 
	 
	 printf("Classifying SCCN with %s\n",ClassifierName[i]);
	 Clustering_Classification(SCCN,PB,TrainingData,ClassifierName[i]);
	 
	 vector<ConfusionMatrix> CM_Temp = GetConfusionMatrix(PB,TrainingData.ClassNumber);
	 
	 
	 
	 char *tempname,*name;
	 name = (char *)malloc(2000*sizeof(char));
	 tempname = (char *)malloc(2000*sizeof(char));
	 
	 tempname = "Clustering_Classification_Result.xls";
	 
	 name = CreateNameIntoFolder(ClassifierFolder,tempname);
	 
	 
	 
	 FILE *res;
	 
	 res = fopen(name,"a+");
	 
	 
	 fprintf(res,"%s\t",substring);
	 for(int j=0;j<CM_Temp.size();j++)
	 {
	   ConfusionMatrix cm_j = CM_Temp[j];
	   fprintf(res,"%d\t%d\t%d\t%d\t%d\t",TrainingData.ClassNumber[j],cm_j.tp,cm_j.fp,cm_j.tn,cm_j.fn);
	   
	   CM_Classifier[i][j].tp = CM_Classifier[i][j].tp + cm_j.tp;
	   CM_Classifier[i][j].fp = CM_Classifier[i][j].fp + cm_j.fp;
	   CM_Classifier[i][j].tn = CM_Classifier[i][j].tn + cm_j.tn;
	   CM_Classifier[i][j].fn = CM_Classifier[i][j].fn + cm_j.fn;
	   
	 }
	 fprintf(res,"\n");
	 fclose(res);

	 CM_Temp.clear();
	 printf("Done\n");
       }
       printf("ClusteringClassification\tDone\n");
     }
      
     
     
     if(SCE_flag)
     {
       clusinfo SCE = ClusteringSCE(PB,ColorTh,SWTh);
       for(int i=0;i<ClassifierName.size();i++)
       {
	 ClassifierFolder = (char *)malloc(2000*sizeof(char));
	 
	 ClassifierFolder = CreateNameIntoFolder(parentfolder,ClassifierName[i]);
	 
	 printf("Classifying SCCN with %s\n",ClassifierName[i]);
	 Clustering_Classification(SCE,PB,TrainingData,ClassifierName[i]);
	 
	 vector<ConfusionMatrix> CM_Temp = GetConfusionMatrix(PB,TrainingData.ClassNumber);
	 
	 
	 
	 char *tempname,*name;
	 name = (char *)malloc(2000*sizeof(char));
	 tempname = (char *)malloc(2000*sizeof(char));
	 
	 tempname = "Clustering_Classification_Result.xls";
	 
	 name = CreateNameIntoFolder(ClassifierFolder,tempname);
	 
	 
	 
	 FILE *res;
	 
	 res = fopen(name,"a+");
	 
	 
	 fprintf(res,"%s\t",substring);
	 for(int j=0;j<CM_Temp.size();j++)
	 {
	   ConfusionMatrix cm_j = CM_Temp[j];
	   fprintf(res,"%d\t%d\t%d\t%d\t%d\t",TrainingData.ClassNumber[j],cm_j.tp,cm_j.fp,cm_j.tn,cm_j.fn);
	   
	   CM_Classifier[i][j].tp = CM_Classifier[i][j].tp + cm_j.tp;
	   CM_Classifier[i][j].fp = CM_Classifier[i][j].fp + cm_j.fp;
	   CM_Classifier[i][j].tn = CM_Classifier[i][j].tn + cm_j.tn;
	   CM_Classifier[i][j].fn = CM_Classifier[i][j].fn + cm_j.fn;
	   
	 }
	 fprintf(res,"\n");
	 fclose(res);

	 CM_Temp.clear();
	 printf("Done\n");
       }
       printf("ClusteringClassification\tDone\n");
     }
     
 }
 

 printf("ALL Input Image classification Completed\n");
 
  char *tempname,*name;
  name = (char *)malloc(2000*sizeof(char));
  tempname = (char *)malloc(2000*sizeof(char));	 
  tempname = "Overall_Clustering_Classification_Result.xls";
  name = CreateNameIntoFolder(parentfolder,tempname);
  
  FILE *res;
  res = fopen(name,"w");
  
  for(int i=0;i<ClassifierName.size();i++)
  {
    MultiClassPerformanceMetrics M;
  
    M.initialize(CM_Classifier[i]);
 
    fprintf(res,"%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",ClassifierName[i],M.GetAverageAccuracy(),M.GetErrorRate(),M.GetPrecesionMu(),M.GetRecallMu(),M.GetFScoreMu(1),M.GetPrecesionM(),M.GetRecallM(),M.GetFScoreM(1));
 
    M.Clear();
  }
  
  fclose(res);

  


  return(0);
}
