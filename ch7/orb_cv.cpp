/**
 * 主要包括如下步骤：
 * 1. 读取图像并初始化
 * 2. 检测 Oriented FAST 角点位置
 * 3. 根据角点位置计算BRIEF算子
 * 4. 对两张图像中的算子进行匹配
 * 5. 对匹配结果进行筛选过滤
 * 6. 绘制匹配结果
 **/

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>  // 2D图像特征检测器和描述子抽取器的抽象基类
#include <opencv2/highgui/highgui.hpp>  // creating and manipulating HighGUI windows and mouse events
#include <chrono>  // 用于处理时间的头文件，所有元素定义在std::chrono命名空间下

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  // 进行输入参数的判断
  if (argc != 3) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }

  //-- 读取图像
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);

  //-- 初始化
  // cv::KeyPoint, 通过2D坐标、领域直径的scale、方向和其它的参数进行定义
  std::vector<KeyPoint> keypoints_1, keypoints_2;  
  // cv::Mat
  Mat descriptors_1, descriptors_2;  

  /**
   * cv::Ptr<T>创建了一个智能指针，类似与std::shared_ptr
   * 
   * cv::ORB 实现了ORB(oriented BRIEF)关键点检测和描述子抽取的类
   * ORB::create()是一个静态公有成员函数，作为ORB类的构造函数，返回一个Ptr<T>指针
   * 
   **/
  Ptr<FeatureDetector> detector = ORB::create();  
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  /**
   * cv:DescriptorMatcher是匹配关键点描述子的抽象基类
   * DescriptorMatcher::create()是一个静态公有成员函数，创建一个指定类型的描述子匹配器
   **/
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

  //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();  // 获取当前时间点
  // 检测关键点，结果存入vector<KeyPoint>中
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  // 计算关键点的描述子，结果存到cv::OutputArray中(Mat可通过转换构造函数得到OutputArray)
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);
  // 计算耗时
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  Mat outimg1;
  // cv::drawKeypoints：画出关键点
  drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  imshow("ORB features", outimg1);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  // cv::DMatch: 用于匹配关键点描述子的类
  // 可以实现: query descriptor index, train descriptor index, train image index, and distance between descriptors.
  vector<DMatch> matches;
  t1 = chrono::steady_clock::now();

  // cv::DescriptorMatcher::match()：从每个描述子中找到最佳匹配
  /**
   * 第一个参数是描述子的query set
   * 第二个参数是描述子的train set
   * 第三个参数是结果存放的位置
   **/
  matcher->match(descriptors_1, descriptors_2, matches);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  //-- 第四步:匹配点对筛选
  // 计算最小距离和最大距离
  // std::minmax_element是C++的STL中的函数，返回指向min和max的一个迭代器pair
  // 第三个参数必须是一个二元函数，接受两个在比较范围内的参数，返回一个可以转换为bool类型的结果
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  // 使用一个经验法则来判断匹配结果
  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  //-- 第五步:绘制匹配结果
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
  imshow("all matches", img_match);
  imshow("good matches", img_goodmatch);
  waitKey(0);

  return 0;
}
