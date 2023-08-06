This directory contains python code from Sarah Song, python notebooks from Steve Richard for learning and a zip archive with a dataset containing mapping from SESAR data to iSamples vocabularies for material, specimentType, sampledFeature, and EarthScience extensions for material and specimentType.  There are 988427 records in the csv dataset. The file size is 490810 kb, too large for standard GitHub tracking, but zips to 35387 Kb, so the zip archive is loaded here in the repo.

fields in the training data:

- doiprefix.   DOI name authority number
- igsn.  IGSN value 
- isgnprefix.  First three letters of the IGSN value
- traintext.  Concatenated text from SESAR database SQL dump dated 2023-04-28.  Carriage returns and line feeds removed. Double quotes removed. These strings include the sample name.
- concatenated_text.    Concatenated text from Sarah Ramdeen training data assembeled For Sarah Song. No sample name, but has registrant, mostly duplicates what's in traintext.
- iSample Material.  isamples material type term. xxx means mapping not applicable. -- generally these are test records, or registration of sampling feature or observations, not physical samples.
- extMaterialType. iSamples earth science extension material type, or mineral group extension type, where applicable. 
- extProtolithMaterialType.  If the extMaterialType is metamorphic or otherwise derived from some protolith this is the protolith. Sparsely populated. 
- iSamMaterialSampleType.  iSamples material sample type (specimen type)
- ExtSampleType. Earth Science extension material sample type, where applicable.
- iSamSampledFeature.  iSamples sampled feature type.

blanks in the extension vocabulary indicates no corresponding extension concept. 
