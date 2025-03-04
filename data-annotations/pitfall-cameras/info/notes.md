
Renamed the csv-file "30-06-22 NARS-30 flash off_csv(1).csv" to "300620 NARS30 Flash off_csv.csv" to use in the dataset. This is the only annotations for WWH fields (LG-WWH-NARS30). There is a "skip" in the annotations compared to the uploaded files, i.e. there are some annotations missing for a range of the images in the "middle" of the folder. It's okay in terms of our dataset though, because these images will simply not be recorded in the COCO-JSON annotations.

Renamed the csv-file "150620 LSE3 Flash on.csv" to "150620 LS3E Flash on.csv" - the LSE3 was clearly a type for the correct camera name, which is LS3E. 

The originals-converted annotations have filenames that match the prefix used for the files they are annotating.

All the images combined are 146 GB

Perhaps we should remove parasitoid? This class covers families across many different orders (e.g. they can both be wasps or mites, ...), so it might not be helpful for a model like this 