// Class: ReadPyKeras
// Automatically generated by MethodBase::MakeClass
//

/* configuration options =====================================================

#GEN -*-*-*-*-*-*-*-*-*-*-*- general info -*-*-*-*-*-*-*-*-*-*-*-

Method         : PyKeras::PyKeras
TMVA Release   : 4.2.1         [262657]
ROOT Release   : 6.15/01       [397057]
Creator        : rajat
Date           : Thu Aug 30 13:25:44 2018
Host           : Linux ehep-TIFR 4.15.0-33-generic #36-Ubuntu SMP Wed Aug 15 16:00:05 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux
Dir            : /home/rajat/github/DNN_practice/DNN-test
Training events: 23000
Analysis type  : [Classification]


#OPT -*-*-*-*-*-*-*-*-*-*-*-*- options -*-*-*-*-*-*-*-*-*-*-*-*-

# Set by User:
V: "False" [Verbose output (short form of "VerbosityLevel" below - overrides the latter one)]
H: "False" [Print method-specific help message]
FilenameModel: "model.h5" [Filename of the initial Keras model]
BatchSize: "200" [Training batch size]
NumEpochs: "10000" [Number of training epochs]
# Default:
VerbosityLevel: "Default" [Verbosity level]
VarTransform: "None" [List of variable transformations performed before training, e.g., "D_Background,P_Signal,G,N_AllClasses" for: "Decorrelation, PCA-transformation, Gaussianisation, Normalisation, each for the given class of events ('AllClasses' denotes all events of all classes, if no class indication is given, 'All' is assumed)"]
CreateMVAPdfs: "False" [Create PDFs for classifier outputs (signal and background)]
IgnoreNegWeightsInTraining: "False" [Events with negative weights are ignored in the training (but are included for testing and performance evaluation)]
FilenameTrainedModel: "dataset/weights/TrainedModel_PyKeras.h5" [Filename of the trained output Keras model]
Verbose: "1" [Keras verbosity during training]
ContinueTraining: "False" [Load weights from previous training]
SaveBestOnly: "True" [Store only weights with smallest validation loss]
TriesEarlyStopping: "-1" [Number of epochs with no improvement in validation loss after which training will be stopped. The default or a negative number deactivates this option.]
LearningRateSchedule: "" [Set new learning rate during training at specific epochs, e.g., "50,0.01;70,0.005"]
TensorBoard: "" [Write a log during training to visualize and monitor the training performance with TensorBoard]
##


#VAR -*-*-*-*-*-*-*-*-*-*-*-* variables *-*-*-*-*-*-*-*-*-*-*-*-

NVar 12
avg_dr_jet                    avg_dr_jet                    avg_dr_jet                    avg_dr_jet                                                      'F'    [0.633560180664,4.03213119507]
dr_lep2_tau_ss                dr_lep2_tau_ss                dr_lep2_tau_ss                dr_lep2_tau_ss                                                  'F'    [0.300122857094,5.28733968735]
mT_lep2                       mT_lep2                       mT_lep2                       mT_lep2                                                         'F'    [0,497.234008789]
mTauTauVis                    mTauTauVis                    mTauTauVis                    mTauTauVis                                                      'F'    [6.79899024963,1167.3807373]
mbb_loose                     mbb_loose                     mbb_loose                     mbb_loose                                                       'F'    [-1,2521.50927734]
mindr_lep1_jet                mindr_lep1_jet                mindr_lep1_jet                mindr_lep1_jet                                                  'F'    [0.400259554386,3.70439887047]
mindr_lep2_jet                mindr_lep2_jet                mindr_lep2_jet                mindr_lep2_jet                                                  'F'    [0.400006175041,3.61695337296]
mindr_tau_jet                 mindr_tau_jet                 mindr_tau_jet                 mindr_tau_jet                                                   'F'    [0.400059759617,3.91682362556]
ptmiss                        ptmiss                        ptmiss                        ptmiss                                                          'F'    [1.33424890041,925.62298584]
tau_eta                       tau_eta                       tau_eta                       tau_eta                                                         'F'    [-2.29905700684,2.29886484146]
tau_pt                        tau_pt                        tau_pt                        tau_pt                                                          'F'    [20.0003414154,579.594238281]
nJet                          nJet                          nJet                          nJet                                                            'F'    [3,13]
NSpec 0


============================================================================ */

#include <array>
#include <vector>
#include <cmath>
#include <string>
#include <iostream>

#ifndef IClassifierReader__def
#define IClassifierReader__def

class IClassifierReader {

 public:

   // constructor
   IClassifierReader() : fStatusIsClean( true ) {}
   virtual ~IClassifierReader() {}

   // return classifier response
   virtual double GetMvaValue( const std::vector<double>& inputValues ) const = 0;

   // returns classifier status
   bool IsStatusClean() const { return fStatusIsClean; }

 protected:

   bool fStatusIsClean;
};

#endif

class ReadPyKeras : public IClassifierReader {

 public:

   // constructor
   ReadPyKeras( std::vector<std::string>& theInputVars )
      : IClassifierReader(),
        fClassName( "ReadPyKeras" ),
        fNvars( 12 )
   {
      // the training input variables
      const char* inputVars[] = { "avg_dr_jet", "dr_lep2_tau_ss", "mT_lep2", "mTauTauVis", "mbb_loose", "mindr_lep1_jet", "mindr_lep2_jet", "mindr_tau_jet", "ptmiss", "tau_eta", "tau_pt", "nJet" };

      // sanity checks
      if (theInputVars.size() <= 0) {
         std::cout << "Problem in class \"" << fClassName << "\": empty input vector" << std::endl;
         fStatusIsClean = false;
      }

      if (theInputVars.size() != fNvars) {
         std::cout << "Problem in class \"" << fClassName << "\": mismatch in number of input values: "
                   << theInputVars.size() << " != " << fNvars << std::endl;
         fStatusIsClean = false;
      }

      // validate input variables
      for (size_t ivar = 0; ivar < theInputVars.size(); ivar++) {
         if (theInputVars[ivar] != inputVars[ivar]) {
            std::cout << "Problem in class \"" << fClassName << "\": mismatch in input variable names" << std::endl
                      << " for variable [" << ivar << "]: " << theInputVars[ivar].c_str() << " != " << inputVars[ivar] << std::endl;
            fStatusIsClean = false;
         }
      }

      // initialize min and max vectors (for normalisation)
      fVmin[0] = 0;
      fVmax[0] = 0;
      fVmin[1] = 0;
      fVmax[1] = 0;
      fVmin[2] = 0;
      fVmax[2] = 0;
      fVmin[3] = 0;
      fVmax[3] = 0;
      fVmin[4] = 0;
      fVmax[4] = 0;
      fVmin[5] = 0;
      fVmax[5] = 0;
      fVmin[6] = 0;
      fVmax[6] = 0;
      fVmin[7] = 0;
      fVmax[7] = 0;
      fVmin[8] = 0;
      fVmax[8] = 0;
      fVmin[9] = 0;
      fVmax[9] = 0;
      fVmin[10] = 0;
      fVmax[10] = 0;
      fVmin[11] = 0;
      fVmax[11] = 0;

      // initialize input variable types
      fType[0] = 'F';
      fType[1] = 'F';
      fType[2] = 'F';
      fType[3] = 'F';
      fType[4] = 'F';
      fType[5] = 'F';
      fType[6] = 'F';
      fType[7] = 'F';
      fType[8] = 'F';
      fType[9] = 'F';
      fType[10] = 'F';
      fType[11] = 'F';

      // initialize constants
      Initialize();

   }

   // destructor
   virtual ~ReadPyKeras() {
      Clear(); // method-specific
   }

   // the classifier response
   // "inputValues" is a vector of input values in the same order as the
   // variables given to the constructor
   double GetMvaValue( const std::vector<double>& inputValues ) const override;

 private:

   // method-specific destructor
   void Clear();

   // common member variables
   const char* fClassName;

   const size_t fNvars;
   size_t GetNvar()           const { return fNvars; }
   char   GetType( int ivar ) const { return fType[ivar]; }

   // normalisation of input variables
   double fVmin[12];
   double fVmax[12];
   double NormVariable( double x, double xmin, double xmax ) const {
      // normalise to output range: [-1, 1]
      return 2*(x - xmin)/(xmax - xmin) - 1.0;
   }

   // type of input variable: 'F' or 'I'
   char   fType[12];

   // initialize internal variables
   void Initialize();
   double GetMvaValue__( const std::vector<double>& inputValues ) const;

   // private members (method specific)
   inline double ReadPyKeras::GetMvaValue( const std::vector<double>& inputValues ) const
   {
      // classifier response value
      double retval = 0;

      // classifier response, sanity check first
      if (!IsStatusClean()) {
         std::cout << "Problem in class \"" << fClassName << "\": cannot return classifier response"
                   << " because status is dirty" << std::endl;
         retval = 0;
      }
      else {
            retval = GetMvaValue__( inputValues );
      }

      return retval;
   }
