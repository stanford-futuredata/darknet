#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <iostream>

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "region_layer.h"

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/videoio/videoio_c.h"
image get_image_from_stream(CvCapture *cap);


using std::string;

const static float kThresh_ = 0.2;
const static float kNms_ = 0.4;
const static int kClasses_ = 80;

class YOLO {
  /*
   * person = 0
   * car = 2
   * bus = 5
   */
 public:
  YOLO(string cfgfile, string weightfile, const int kTargetClass) :
      kTargetClass_(kTargetClass),
      net_(parse_network_cfg(const_cast<char *>(cfgfile.c_str()))),
      l_(net_.layers[net_.n - 1]),
      boxes_(l_.w * l_.h * l_.n),
      probs_(l_.w * l_.h * l_.n) {
    // net_ = parse_network_cfg(const_cast<char *>(cfgfile.c_str()));
    load_weights(&net_, const_cast<char *>(weightfile.c_str()));

    // FIXME: more batches?
    set_batch_network(&net_, 1);
    layer l = net_.layers[net_.n - 1];
    if (l.classes != kClasses_) {
      std::cerr << l.n << "\n";
      throw std::runtime_error("YOLO network has wrong # of classes");
    }
    srand(2222222);

    for (int j = 0; j < l.w * l.h * l.n; j++)
      probs_[j] = (float *) calloc(kClasses_, sizeof(float));
  }

  float LabelFrame(image im) {
    image sized = resize_image(im, net_.w, net_.h);
    float *X = sized.data;
    layer l = net_.layers[net_.n - 1];

    network_predict(net_, X);
    if (l.type == DETECTION) {
      get_detection_boxes(l, 1, 1, kThresh_, &probs_[0], &boxes_[0], 0);
    } else if (l.type == REGION) {
      get_region_boxes(l, 1, 1, kThresh_, &probs_[0], &boxes_[0], 0, 0, kThresh_);
    } else {
      throw std::runtime_error("YOLO last error must be detections");
    }
    if (kNms_ > 0) do_nms(&boxes_[0], &probs_[0], l.w*l.h*l.n, kClasses_, kNms_);

    float max_prob = 0;
    for (int j = 0; j < l.w * l.h * l.n; j++) {
      const float confidence = probs_[j][kTargetClass_];
      if (confidence > max_prob)
        max_prob = confidence;
    }
    return max_prob;
  }

 private:
  constexpr static float kThresh_ = 0.2;
  constexpr static float kNms_ = 0.4;
  constexpr static int kClasses_ = 80;  // Maybe fixme?
  const int kTargetClass_;

  network net_;
  layer l_;

  std::vector<box> boxes_;
  std::vector<float *> probs_;
};

/*int main(int argc, char **argv) {
  int i;
  // FIXME
  char cfg_file[] = "cfg/yolo.cfg";
  char weight_file[] = "weights/yolo.weights";

  // Init yolo
  YOLO yolo = YOLO(cfg_file, weight_file, 0);

  CvCapture *cap = cvCaptureFromFile("/dfs/scratch1/fabuzaid/vuse-datasets-completed/videos/gates-elevator.mp4");

  // Label images
  i = 0;
  while (++i >= 0) {
    if (i % 500 == 0)
      std::cout << "frame " << i << "\n";
    if (i >= 2000)
      break;
    image im = get_image_from_stream(cap);
    if (!im.data) {
      std::cerr << "Stream closed.\n";
      break;
    }
    std::cout << i << ": " << yolo.LabelFrame(im) << "\n";
    free_image(im);
  }

  // cvReleaseCapture(cap);
  return 0;
}*/
