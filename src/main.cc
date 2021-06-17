#include <CLI/CLI.hpp>

#include "../include/lbp.hh"


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
   cv::Mat labels_mat;
   if (mode == "CPU")
   {
     labels_mat = cpu_lbp(inputfilename);
   }
   else if (mode == "GPU")
   {
     labels_mat = gpu_lbp(inputfilename);
   }
   else
   {
       printf("Invalid argument");
       return 1;
   }

   // Save
   cv::imwrite(filename, labels_mat);
   printf("Output saved in %s.", filename);
   return 0;
 }
  
