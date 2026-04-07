# September 15, 2025
# Ian Dryg
# Dana-Farber Cancer Institute
# Center for Immuno-Oncology
# This code is used to VALIDATE outputs from Lunaphore's Horizon analysis software for use with pythologist. 
# It quickly checks the output from Horizon and flags any issues before we proceed to data ingestion. 

# Nested Annotation Strategy
# Uses nested annotations indicated by a naming protocol in the Annotation Group column. 
# [annotation type]_[main id].[roi id].[exclusion id]_[shape type]
# Main_1.0.0_Rectangle
# ROI_1.1.0_Polygon - the first ROI within the Main 1 annotation
# ROI_1.2.0_Polygon - the second ROI within the Main 1 annotation
# Exclusion_1.1.1_Polygon - the first Exclusion within ROI 1 within Main 1

import pandas as pd
import numpy as np
import os
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


# validate horizon exports before ingesting. 
# horizon_export_path: path to input data file from horizon export. Must be a string filepath. This contains areas and cells that need to be parsed. 
# Note: in this version, if the cells_df_path and area_df_path are the same, the program assumes they're combined in the same file and will split them. 
# savefile_dir: path to the directory you want to save the outputs in. 
# microns_per_pixel: should be 0.28 for the Lunaphore instrument as of 20241220
def validate_lunaphore(horizon_export_filepath,
                        savefile_dir,
                        microns_per_pixel=0.28):
    
    # validate input parameters
    try:
        validate_parameters(horizon_export_filepath, savefile_dir, microns_per_pixel)
    except (TypeError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print(str(horizon_export_filepath))
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")

    # horizon_export_filepath: one input file containing the area of the Main annotation, along with the cells in that Main annotation. 
    # Other annotation areas may be in this file as well. 
    # There could be multiple annotations. We'll handle them all using a dictionary. 
    
    # read in the horizon output
    temp_input_df1 = pd.read_csv(horizon_export_filepath, low_memory=False)

    # remove unwanted column
    if 'Unnamed: 0' in list(temp_input_df1.columns):
        temp_input_df1 = temp_input_df1.drop(columns=['Unnamed: 0'])
        
    # cell annotations have Nuclei Segmentation at the end of the Annotation Group "path", label those rows
    temp_input_df1.loc[temp_input_df1['Annotation Group'].str.contains('Nuclei Segmentation'), 'annot_name'] = 'Nuclei Segmentation'
    # area annotations have the annotation name at the end of the Annotation Group "path". Extract those names
    temp_input_df1.loc[~temp_input_df1['Annotation Group'].str.contains('Nuclei Segmentation'), 'annot_name'] = temp_input_df1['Annotation Group'].str.split('/').str[-1]
    # Split the cells and the areas dfs apart. 
    # cells df has Nuclei Segmentation in the Annotation Group "path", extract those rows
    temp_input_cells = temp_input_df1.loc[temp_input_df1['Annotation Group'].str.contains('Nuclei Segmentation')]
    # area df does not have Nuclei Segmentation in the Annotation Group "path", extract those rows
    temp_input_area = temp_input_df1.loc[~temp_input_df1['Annotation Group'].str.contains('Nuclei Segmentation')]
    
    # remove Exclusions from cells dataframe if there are any
    if 'Class group' in list(temp_input_cells.columns):
        print('Removing exclusions...')
        print('Size before removing exclusions: ' + str(temp_input_cells.shape))
        temp_input_cells = temp_input_cells.loc[temp_input_cells['Class group']!='Exclusions']
        print('Size after removing exclusions: ' + str(temp_input_cells.shape))

    # Label parent annotations for CELLS (Main and ROI). Should be in Class group column. 
    temp_input_cells['parent_id'] = temp_input_cells['Class group'].str.lower().str.split('_').str[1]
    temp_input_cells['parent_main_id'] = temp_input_cells['Class group'].str.lower().str.split('_').str[1].str.split('.').str[0]
    temp_input_cells['parent_roi_id'] = temp_input_cells['Class group'].str.lower().str.split('_').str[1].str.split('.').str[1]
    
    # drop columns that only contain nan
    temp_input_cells = temp_input_cells.dropna(axis=1, how='all')
    temp_input_area = temp_input_area.dropna(axis=1, how='all')

    # Label annotation info for all annotations
    # ex: ROI_1.1.0_Polygon
    temp_input_area['annot_type'] = temp_input_area['annot_name'].str.lower().str.split('_').str[0]
    temp_input_area['annot_shape'] = temp_input_area['annot_name'].str.lower().str.split('_').str[2]
    temp_input_area['full_annot_id'] = temp_input_area['annot_name'].str.lower().str.split('_').str[1]
    temp_input_area['main_annot_id'] = temp_input_area['annot_name'].str.lower().str.split('_').str[1].str.split('.').str[0]
    temp_input_area['roi_annot_id'] = temp_input_area['annot_name'].str.lower().str.split('_').str[1].str.split('.').str[1]
    temp_input_area['exclusion_annot_id'] = temp_input_area['annot_name'].str.lower().str.split('_').str[1].str.split('.').str[2]

    print('area df: ')
    display(temp_input_area)

    # ---------------------------------------------------------
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("Check cells (excluded cells were dropped already):")
    print(temp_input_cells['Class group'].value_counts())
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("Review area annotations:")
    print(temp_input_area['full_annot_id'])
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("Check annotations for required columns:")
    areas_req_cols = ['Annotation Group','Area in μm²','X Position in μm','Y Position in μm']
    missing_area_cols = [x for x in areas_req_cols if x not in temp_input_area.columns]
    if len(missing_area_cols) > 0:
        print('Annotations are missing the following columns:')
        print(missing_area_cols)
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("Check if annotations are missing fields: ")
    print("Number of NaNs in required columns: ")
    for curr_col in areas_req_cols:
        if curr_col in temp_input_area.columns:
            print(curr_col)
            num_nans = temp_input_area[curr_col].isna().sum()
            print(str(num_nans))
            if num_nans > 0:
                print("Annotations with NaNs: ")
                print(temp_input_area.loc[temp_input_area[curr_col].isna()]['Annotation Group'])
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("view areas df: ")
    display(temp_input_area)
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("Check cells for duplicate thresholds: ")
    # ---------------------------------------------------------
    # check meta on cells data, drop duplicate markers if necessary
    qc_input_cells = extract_column_metadata(temp_input_cells)
    # check for duplicate thresholds per marker in the cell compartment
    qc_input_cells_cell_thresholds = qc_input_cells.loc[(qc_input_cells['Measurement_Type']=='Threshold') & (qc_input_cells['Compartment_Type']=='Cell')]
    # keep duplicates AND originals to investigate
    qc_input_cells_threshold_duplicates = qc_input_cells_cell_thresholds.loc[qc_input_cells_cell_thresholds.duplicated(subset=['Label_Mapping'], keep=False)]
    # Warning and display if there are ny duplicates. 
    if not qc_input_cells_threshold_duplicates.empty:
        unique_labels = qc_input_cells_threshold_duplicates['Label_Mapping'].unique()
        warnings.warn(
            f"---------------------------------------------------------\n"
            f"Duplicate thresholds found for the following markers: \n"
            f"Duplicated Label_Mapping values: {list(unique_labels)}\n"
            f"---------------------------------------------------------\n"
        )
        display(qc_input_cells_threshold_duplicates)
        warnings.warn(f"See warning above. When running the ingestion, please input a list of integers to choose_duplicate_threshold_to_keep specifying which indexes in the duplicated dataframe above to KEEP. Others will be dropped. \n")
    # ---------------------------------------------------------
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")


# function to validate input parameters before proceeding
def validate_parameters(horizon_export_filepath, savefile_dir, microns_per_pixel):
    # Type checking
    if not isinstance(horizon_export_filepath, str):
        raise TypeError("horizon_export_filepath must be a string")
    if not isinstance(savefile_dir, str):
        raise TypeError("savefile_dir must be a string")
    if not isinstance(microns_per_pixel, float):
        raise TypeError("microns_per_pixel must be a float")
    # validate files exist
    if not os.path.isfile(horizon_export_filepath):
        raise ValueError(f"The path '{horizon_export_filepath}' is not a file")


# function to integrate region areas into the cells dataframe. 
# regions will just be called ANY (assuming there's only one)
def add_region_areas(cells_df, area_df, microns_per_pixel):
    # Set region label to ANY
    area_df['region_label'] = 'ANY'
    cells_df['region_label'] = 'ANY'
    # convert areas from microns into pixels, which is expected by pythologst. 
    # area_pixles2 = area_microns2*((1/microns_per_pixel)**2)
    area_df['Area_pixels_squared'] = area_df['Area in μm²'].apply(lambda x: round(x*((1/microns_per_pixel)**2), 3))
    # Extract dictionary of region areas using ANY as region label
    region_areas_dict = dict(zip(area_df['region_label'],area_df['Area_pixels_squared']))
    # put this region areas dict into the regions column. It'll be the same value for every row. 
    cells_df['regions'] = cells_df['region_label'].apply(lambda x: region_areas_dict)
    
    return cells_df


# function to extract metadata
# This function labels columns in the dataset to indicate which columns should be used for what later on. 
# It also helps set up to rename the columns later. 

def extract_column_metadata(cells_df):
    import numpy as np
    
    # extract columns
    df = pd.DataFrame({'orig_cols':cells_df.columns})
    
    # define key measurements to reformat
    measurement_type_map = {
        'Annotation Group':'Annotation Group',
        'Annotation Index':'cell_index',
        'X Position':'x',
        'Y Position':'y',
        'Area':'cell_area',
        'Mean Intensity':'Mean Intensity',
        'Threshold':'Threshold',
        'regions':'regions',
        'region_label':'region_label',
        'parent_id':'Parent Annotation',
        'Leiden clusters':'Leiden clusters'
    }
    
    compartment_type_map = {
        'Cells':'Cell',
        'Cytoplasm':'Cytoplasm',
        'Nuclei':'Nucleus',
        'Annotation':'Annotation',
        'Position':'Annotation',
        'region':'Annotation',
        'parent_id':'Annotation',
        'Leiden clusters':'Annotation'
    }
    
    # ----------------------------
    # Extract Measurement Type
    df['Measurement_Type'] = ''
    for k,v in measurement_type_map.items():
        df.loc[df['orig_cols'].str.contains(k), 'Measurement_Type'] = v
    
    # ----------------------------
    # Extract Compartment Type
    df['Compartment_Type'] = ''
    for k,v in compartment_type_map.items():
        df.loc[df['orig_cols'].str.contains(k), 'Compartment_Type'] = v
    
    # ----------------------------
    # Extract Marker Label
    temp_marker_df = df['orig_cols'].str.split('(', expand=True)
    # use  3rd column of output to separate further, rows without values should be None...
    temp_marker_df2 = temp_marker_df.iloc[:, 2].str.split(' - ', expand=True)
    # Nucleus has slightly different structure. Marker is right after first (, not second. 
    # use  2nd column of output to separate further, rows without values should be None...
    temp_marker_df2_nucleus = temp_marker_df.iloc[:, 1].str.split(' - ', expand=True)
    # conditionally fill where it's Nucleus and Mean Intensity using nucleus outputs, otherwise use other outputs. 
    # use 1st column to separate out the marker
    df['Marker'] = np.where((df['Compartment_Type']=='Nucleus') & (df['Measurement_Type']=='Threshold'), 
                            temp_marker_df2_nucleus.iloc[:, 0], 
                            temp_marker_df2.iloc[:, 0])
    # manually edit DAPI
    df.loc[df['orig_cols'].str.contains('DAPI'),'Marker'] = 'DAPI'
    # get just the first part of Marker...
    # some don't already have _ so we add one just in case
    df['Marker'] = df['Marker'] + '_'
    temp_marker_df_first = df['Marker'].str.split('_', expand=True)
    df['Marker'] = temp_marker_df_first.iloc[:, 0]
    # Remove hyphens
    df['Marker'] = df['Marker'].str.replace('-', '')
    
    # ----------------------------
    # Extract Channel Label
    # use temp_marker_df2 for most fields
    temp_marker_df3 = temp_marker_df2.iloc[:, 1].str.split(')', expand=True)
    # use temp_marker_df2_nucleus for nuclei
    if temp_marker_df2_nucleus.shape[1] > 1:
        # if there are nuclei...
        temp_marker_df3_nucleus = temp_marker_df2_nucleus.iloc[:, 1].str.split(')', expand=True)
    else:
        # if there aren't nuclei... (turn series into dataframe)
        temp_marker_df3_nucleus = pd.DataFrame(temp_marker_df2_nucleus.iloc[:,0].apply(lambda x: np.nan))
    # conditionally fill where it's Nucleus and Mean Intensity
    df['Channel'] = np.where((df['Compartment_Type']=='Nucleus') & (df['Measurement_Type']=='Threshold'),
                             temp_marker_df3_nucleus.iloc[:, 0],
                             temp_marker_df3.iloc[:, 0])
    # manually edit DAPI
    df.loc[df['orig_cols'].str.contains('DAPI'),'Channel'] = 'DAPI'
    
    # ----------------------------
    # Extract Thresholds
    # same for nuclei and other cell compartments
    temp_thresholds_df = df['orig_cols'].str.split('Threshold \\(', expand=True)
    # Take 2nd column, after the "Threshold ("
    temp_thresholds_df2 = temp_thresholds_df.iloc[:, 1].str.split(')', expand=True)
    # Take 1st column, before the ")"
    df['Threshold'] = temp_thresholds_df2.iloc[:, 0]
    
    # ----------------------------
    # Edge cases:
    # manually set the Measurement Type for Area Threshold and Position Thresholds
    df.loc[(df['orig_cols'].str.contains('Area')) & (df['orig_cols'].str.contains('Threshold')),'Measurement_Type'] = 'Area Threshold'
    df.loc[(df['orig_cols'].str.contains('Position')) & (df['orig_cols'].str.contains('Threshold')),'Measurement_Type'] = 'Position Threshold'
    # By default let's use x and y positions of Cell (should be same as nuclei anyway)
    # so let's set Measurement_Type of the nucelei x and y to something specific...
    df.loc[(df['orig_cols'].str.contains('X Position')) & (df['orig_cols'].str.contains('Nuclei')),'Measurement_Type'] = 'nucleus_x'
    df.loc[(df['orig_cols'].str.contains('Y Position')) & (df['orig_cols'].str.contains('Nuclei')),'Measurement_Type'] = 'nucleus_y'
    # manually set Marker name to nan for these
    df.loc[(df['orig_cols'].str.contains('Area')) & (df['orig_cols'].str.contains('Threshold')),'Marker'] = np.nan
    df.loc[(df['orig_cols'].str.contains('Position')) & (df['orig_cols'].str.contains('Threshold')),'Marker'] = np.nan
    
    # ----------------------------
    # Make a column defining renaming:
    # use markers 
    df['Label_Mapping'] = df['Marker']
    # if markers are na, use measurement type
    df.loc[df['Label_Mapping'].isna(), 'Label_Mapping'] = df['Measurement_Type']
    
    # return column metadata
    return df


# function to check Marker names
def check_markers(meta):
    display(list(meta['Marker'].dropna().unique()))


# function to rename Marker names
def rename_marker(meta, old_marker_name, new_marker_name):
    meta['Marker'] = meta['Marker'].replace(old_marker_name, new_marker_name)
    # also update Label_Mapping column
    meta['Label_Mapping'] = meta['Label_Mapping'].replace(old_marker_name, new_marker_name)
    return meta
