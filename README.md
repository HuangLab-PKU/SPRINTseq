# SPRINT-seq

SPRINTseq (Spatially Resolved and signal-diluted Next-generation Targeted sequencing) is an innovative in situ sequencing strategy that combines hybrid block coding and molecular dilution strategies, allowing an accurate RNA detection in crowded environment. This method can profile a mouse brain coronal slice within 1 day, and generate gene expression architecture in a sub-micron resolution.

For more information, please read the article.  [Chang et.al (2023) *bioRxiv*](https://doi.org/10.1101/2022.11.16.516714)

# Code Preview

Code for SPRINT-seq consists of five parts, **barcode_design**, **image_processing**, **gene_calling** **cell_segmentation** and **expression_matrix_analysis**. Data will be processed in this order.

# Data Architecture

Raw data base directory and processed data output directory can be whatever place you need. But its subdirectory should be like this:

Raw data root                           

|---RUN_ID1

|---RUN_ID2

|...

|---RUN_IDN



Output root

|---RUN_ID1_processed (automate created)

|    |---focal_stacked (automate created)

|     |---background_corrected (automate created)

|    |---registered (automate created)

|    |---stitched (automate created)

|    |---readout (automate created)

|    |---segmented (automate created)

Your raw data should be in folder RUN_ID.

# Start

This code should be run under `python 3.8`. Later version will bring some environment problems.

To run this code, packages must be installed with command:

```shell
pip install -r requirements.txt
```

And MATLAB engine should be installed, according to your local MATLAB version, you can follow [official guideline](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).  

For example, MATLAB R2021b：`python -m pip install matlabengine==9.11.21`

And **pip setuptools** needs a correct version. 

For MATLAB R2021b: `pip install --upgrade pip setuptools==57.5.0`

**Remind!** There are lots of paths or directories need to be edited in files mentioned below.

## Barcode Design

Code in this part will generate a suitable barcode space for you to choose. 

**Caution!** These codes may cost a lot of memory.

Step 1: Run `generate_graph.py`.  An in-between file will be generated.

Step 2: Run `MIS_variance_trail_qsub.py` using the in-between file as input. Suitable barcodes will be generated in a txt file.

We have provided example in-between file `test_graph_210809_G58.txt` and example final file `min_var_mis_0809_G58.txt`. 

## Image Processing

Step 1: Edit the directory in python file `scan_fstack.py` as the directory you wish. Run the code: 

```shell
python scan_fstack.py Raw_data_root
```

We have provided a preprocessed example data for Step 2 and pipeline after, which path is `./example_dataset`.  You should create a folder `output_root/whatever_name_processed/focal_stacked/` and put all example data in it.

Step 2: Edit the directory in python file `image_process_after_stack.py` the same as the directory before. Run the code: 

```shell
python image_proess_after_stack.py
```

Results will be large stitched images, which will be used in next part.

## Gene Calling

Step 1:  Before running code, a padlock sequence file should be provided. It should be a csv file and formatted like:

```csv
Name,Padlock
Gene_name1,Padlock_sequence1
Gene_name2,Padlock_sequence2
...,...
Gene_nameN,Padlock_sequenceN
```

Step 2: Run the jupyter notebook file `generate_ref_files.ipynb`. A reference barcode book will be produced named `Your_Given_Name.csv`.

Step 3: Edit the directory in `Gene_calling.py` and run it. A csv file containing spots coordinate and gene name will be generated, named `output_root/whatever_name_processed/readout/mapped_genes.csv`.

We have provided example csv file `108_plex_10base_barcodes_padlock_sequence.csv` and example barcode book `plex_map_filtered_108plex_10base_barcode.csv`. 

## Cell Segmentation

Edit the directory in `segment.py` and run it. This code will segment cell nucleus according to DAPI channel. A csv file containing the coordinate of nucleus centroid will be generated named `output_root/whatever_name_processed/segmented/centroids_all.csv`.

## Expression Matrix Analysis

Edit the directory and run the jupyter notebook file `integrated_analysis_SPRINTseq.ipynb`, important post processing results will be shown inside the jupyter.

The provided dataset is just an example data, so we provided another true experimental data for post-analysis in folder `example_dataset_whole_brain`. Run `merge.py` to build dataset from splited dataset, and replace `mapped_genes.csv` and `centroids_all.csv` mentioned above with file in this folder to use this dataset for Matrix Analysis.
