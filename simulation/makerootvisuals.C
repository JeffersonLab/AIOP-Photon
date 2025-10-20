#include <iostream>
#include <TROOT.h>
void makerootvisuals() {
   std::cout << "Building the C++ rootvisuals_C.so and supporting libraries" << std::endl
             << "for the spotfinder tool. Make sure that LD_PRELOAD is not" << std::endl
             << "set in your environment, or you are in for a bad surprise!" << std::endl;
   gSystem->AddIncludePath("-I/usr/include/python3.6m");
   gSystem->AddIncludePath("-I./Analysis");
   gSystem->AddIncludePath("-I.");
   gSystem->Load("libboost_python3.so");
   gSystem->CompileMacro("Map2D.cc", "kg");
   gSystem->CompileMacro("Couples.C", "kg");
   gSystem->AddIncludePath("-I/cvmfs/oasis.opensciencegrid.org/gluex/Diracxx/x86_64/include");
   gSystem->CompileMacro("CobremsGeneration.cc", "kg");
   gSystem->CompileMacro("rootvisuals.C", "kg");
}
