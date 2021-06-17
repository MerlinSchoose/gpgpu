#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include "lbp.hh"
#include "gpu_lbp.hh"
#include "utils.hh"


/*
 * ./exe -m GPU 
 * ./exe -m CPU
 */
int main(int argc, char** argv)
 {
   (void) argc;
   (void) argv;
  
   std::string filename = "../results/output.png";
   std::string inputfilename = "../data/barcode-00-01.jpg";
   std::string mode = "GPU";
  
   CLI::App app{"gpgpu"};
   app.add_option("-i", inputfilename, "Input image");
   app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");
  
   CLI11_PARSE(app, argc, argv);
  
   // Rendering
   cv::Mat image = cv::imread(inputfilename,
       cv::IMREAD_GRAYSCALE);


   cv::Mat labels_mat;
   if (mode == "CPU")
   {
     labels_mat = cpu_lbp(image);
   }
   else if (mode == "GPU")
   {
     labels_mat = vect_to_mat(gpu_lbp(mat_to_vect(image)));
   }
   else
   {
     spdlog::info("Invalid argument");
       return 1;
   }

   // Save
   cv::imwrite(filename, labels_mat);
   spdlog::info("Output saved in {}.", filename);
   return 0;
 }
  
