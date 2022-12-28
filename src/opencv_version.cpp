#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
  cout << "OpenCV version : " << CV_VERSION << endl;
  // cout << "Major version : " << CV_MAJOR_VERSION << endl;
  // cout << "Minor version : " << CV_MINOR_VERSION << endl;
  // cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;

  if ( CV_MAJOR_VERSION < 3)
  {
    // Old OpenCV 2
    std::cout << "Hello OPENCV less than 3" << std::endl;
  } 
  else
  {
    // New OpenCV 3
    std::cout << "Hello World CV > 3" << std::endl;
    
    // For more detailed information, run below command:
    // printf("OpenCV: %s", cv::getBuildInformation().c_str());
  }
}

