cmake_minimum_required(VERSION 2.8)
project( Clustering_SCCN_SCE )
set(BUILD_SHARED_LIBS OFF)

find_package( OpenCV REQUIRED )

if (OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})
endif()

SET(Alglib_Source	
			alglib/ap.cpp 
			alglib/alglibinternal.cpp 
			alglib/alglibmisc.cpp 
			alglib/linalg.cpp 
			alglib/specialfunctions.cpp 
			alglib/statistics.cpp
#			alglib/alglibmisc.cpp
#			alglib/ap.cpp
#			alglib/dataanalysis.cpp
#			alglib/diffequations.cpp
#			alglib/fasttransforms.cpp
#			alglib/integration.cpp
#			alglib/interpolation.cpp
#			alglib/linalg.cpp
#			alglib/optimization.cpp
#			alglib/solvers.cpp
#			alglib/specialfunctions.cpp
#			alglib/statistics.cpp
						  )
			
# SET(Alglib_Headers 	
# 			alglib/alglibinternal.h
# 			alglib/alglibmisc.h
# 			alglib/ap.h
# 			alglib/dataanalysis.h
# 			alglib/diffequations.h
# 			alglib/fasttransforms.h
# 			alglib/integration.h
# 			alglib/interpolation.h
# 			alglib/linalg.h
# 			alglib/optimization.h
# 			alglib/solvers.h
# 			alglib/specialfunctions.h
# 			alglib/statistics.h
# 			alglib/stdafx.h)

SET(Clustering_HEADERS 	
				StandardHeaders.h
				sorting.h
				StatisticalFunctions.h
				folder.h
				binarization.h
				PixelValidation.h
				connectedcomponent.h				
				GeneralFunctions.h
				HSV.h
				Image_proc_functions.h
				Morphology.h				
				RectangleTest.h
				ScalarColorFeature.h
				SmoothingGapfilling.h								
				StrokeWidth.h
				Clustering_Classification.h)  


SET(Clustering_SOURCES  	
				Clustering_Classification.cpp
				folder.cpp
				sorting.cpp
				StatisticalFunctions.cpp
				PixelValidation.cpp
				binarization.cpp
				connectedcomponent.cpp
				HSV.cpp
				GeneralFunctions.cpp				
				Image_proc_functions.cpp
				Morphology.cpp				
				RectangleTest.cpp
				ScalarColorFeature.cpp
				SmoothingGapfilling.cpp								
				StrokeWidth.cpp
				)


# Or set it globally for all targets:
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

ADD_EXECUTABLE(Clustering ${Alglib_Source} ${Clustering_HEADERS} ${Clustering_SOURCES})
TARGET_LINK_LIBRARIES(Clustering ${OpenCV_LIBS})
set(CMAKE_BUILD_TYPE Debug)

