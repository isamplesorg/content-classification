This directory contains python code from Sarah Song, python notebooks from Steve Richard for learning and a zip archive with a dataset containing mapping from SESAR data to iSamples vocabularies for material, specimentType, sampledFeature, and EarthScience extensions for material and specimentType.  There are 988427 records in the csv dataset. The file size is 490810 kb, too large for standard GitHub tracking, but zips to 35387 Kb, so the zip archive is loaded here in the repo.

fields in the training data:

- doiprefix.   DOI name authority number
- igsn.  IGSN value 
- isgnprefix.  First three letters of the IGSN value
- traintext.  Concatenated text from SESAR database SQL dump dated 2023-04-28.  Carriage returns and line feeds removed. Double quotes removed. These strings include the sample name. The content of this field has been edited to remove long text strings about sample location (mostly from Adam Mansur) and various other global text edits to shorten the strings and concentrate on information useful for classification.  In the future we'll need to be more considered about what gets concatenated. Currently working with 80 character limit on length of training text character string in the ML code. 
- concatenated_text.    Concatenated text from Sarah Ramdeen training data assembeled For Sarah Song. No sample name, but has registrant, mostly duplicates what's in traintext.
- iSampleMaterial.  isamples material type term. xxx means mapping not applicable. -- generally these are test records, or registration of sampling feature or observations, not physical samples.  'xxx' type is for things that aren't material samples. Also nonaqueousfluid (1 record) and waterice (4 records) are mapped to xxx. don't have enough examples to train.
- iSampleMaterial2.  if the sample maps to multiple material types, put the second type here (none map to 3). Assignme of the '2' material type is arbitrary at this point. Have to figure out how this might be handled.
- extMaterialType. iSamples earth science extension material type, or mineral group extension type, where applicable. 
- extProtolithMaterialType.  If the extMaterialType is metamorphic or otherwise derived from some protolith this is the protolith. Sparsely populated. 
- iSamMaterialSampleType.  iSamples material sample type (specimen type)
- ExtSampleType. Earth Science extension material sample type, where applicable.
- iSamSampledFeature.  iSamples sampled feature type.

blanks in the extension vocabulary indicates no corresponding extension concept. 
currently the ML code (TrainWorkSMR.ipynb) only loads igsn, traintext, and iSampleMaterial into the dataframe for training.
