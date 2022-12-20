# OCT2L

## OCT two layer segmentation

Install required packages:
```
python3 -m pip install -r requirements.txt
``` 

To run:
```
python3 oct_two_layer.py --folder project-files-2022-11-30-npy
```

`oct_two_layer` segments ILM via looking for 'bright spots' and RPE via Canny Edge detector. It looks ok but is not amazing, especially the ILM, but this is where I'm leavin the segmentation code. Working on the mesh construction. 

## OCT ODD-md.AI segmentation

11-Dec-2022 updates:
OCT volumes are exported from the Flywheel platform via `fw_project_tar_get.py`. I exported all files that were either .npy or .E2E. Clinician's segmentations were exported from md.AI via `mdai_labelled_data_get.py` and `mdai_laballed_data_example.ipynb`. Vertices are stored as binary image masks, first in .npy then converted to .png. The data is stored by md.AI identification numbers. 

`mdai-odd-data-wrangling.ipynb`: imports the training data (images and masks)
`mdai_odd_data_wrangling.py`: converts ODD dataset into a torch.utils.data.Dataset, splits data into train-test-val sets.



Currently building a VGG16 neural network. 


