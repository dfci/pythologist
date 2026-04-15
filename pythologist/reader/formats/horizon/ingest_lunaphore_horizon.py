# April 6th, 2026
# Ian Dryg
# Dana-Farber Cancer Institute
# Center for Immuno-Oncology
# This code is used to ingest outputs from Lunaphore's Horizon analysis software for use with pythologist. 
# It converts the output from Horizon into a pythologist CellDataFrame for further use in our pipelines. 

# Uses nested annotations indicated by a naming protocol in the Annotation Group column. 
# [annotation type]_[main id].[roi id].[exclusion id]_[shape type]
# Main_1.0.0_Rectangle
# ROI_1.1.0_Polygon - the first ROI within the Main 1 annotation
# ROI_1.2.0_Polygon - the second ROI within the Main 1 annotation
# Exclusion_1.1.1_Polygon - the first Exclusion within ROI 1 within Main 1

# Changes:
# Update 2026/04/06: Adjust microns to pixel conversion so that is it not hard coded. Also transform microns into pixels

# TODO: process single cell data the same way the cdf is processed
# TODO: are we happy with what intensity value is grabbed for each marker? Default is cell
# TODO: frame dimensions
# TODO: change to merging on stable indexes
# TODO: change microns to pixels conversion to be earlier in the process
# TODO: remove troubleshooting print statements

import pandas as pd
import numpy as np
import uuid
from pythologist import CellDataFrame, SubsetLogic as SL, PercentageLogic as PL
import os
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


# run all the ingestion methods
# horizon_export_path: path to input data file from horizon export. Must be a string filepath. This contains areas and cells that need to be parsed. 
# Note: in this version, if the cells_df_path and area_df_path are the same, the program assumes they're combined in the same file and will split them. 
# project_name: name of the project. Example: 'Nick_Horizon_Testing_20241219'
# savefile_dir: path to the directory you want to save the outputs in. 
# savefile_name: name of the base name for the outputs. There will be a separate output cdf for each annotation. 
# microns_per_pixel: should be 0.28 for the Lunaphore instrument as of 20241220
# run_qc: set to True if you want to check the cdf qc
# return_cdf: set to True if you want to return the cdf, otherwise just saves the file and doesn't return anything
def run_lunaphore_ingestion(horizon_export_filepath,
                            project_name,
                            savefile_dir,
                            savefile_name,
                            overwrite_sample_name = None,
                            default_phenotype = 'CD3', 
                            microns_per_pixel=0.28, #
                            run_qc = False,
                            save_cdf = False,
                            save_full_single_cell = False,
                            return_cdf = False,
                            show_meta = False,
                            rename_markers_dict = None,
                            choose_duplicate_threshold_to_keep = None):
    
    # ---------------------------------------------------------
    # validate input parameters
    try:
        validate_parameters(horizon_export_filepath, project_name, savefile_dir, savefile_name, microns_per_pixel, run_qc, return_cdf, save_full_single_cell, overwrite_sample_name, default_phenotype)
    except (TypeError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    # horizon_export_filepath: one input file containing the area of the Main annotation, along with the cells in that Main annotation. 
    # Other annotation areas may be in this file as well. 
    # There could be multiple annotations. We'll handle them all using a dictionary. 
    
    # ---------------------------------------------------------
    # import horizon file, get cells and area dfs
    temp_input_cells, temp_input_area = import_horizon_file(horizon_export_filepath)

    print('area df: ')
    display(temp_input_area)

    # ---------------------------------------------------------
    # QC check on metadata for cells data.
    # To detect if there are duplicate markers, and drop them if necessary
    qc_input_cells = extract_column_metadata(temp_input_cells)

    # check for duplicate thresholds per marker in the cell compartment
    #qc_input_cells_cell_thresholds = qc_input_cells.loc[(qc_input_cells['Measurement_Type']=='Threshold') & (qc_input_cells['Compartment_Type']=='Cell')]
    qc_input_cells_cell_thresholds = qc_input_cells.loc[qc_input_cells['Measurement_Type']=='Threshold']
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
        if choose_duplicate_threshold_to_keep is None:
            raise ValueError(
            f"See warning above. Please input a list of integers to choose_duplicate_threshold_to_keep specifying which indexes in the duplicated dataframe above to KEEP. Others will be dropped. \n")
        elif type(choose_duplicate_threshold_to_keep) is list:
            # get the opposite of what we want to keep, so we can drop these from the original data columns
            qc_input_cells_threshold_duplicates_to_drop = qc_input_cells_threshold_duplicates.drop(choose_duplicate_threshold_to_keep)
            drop_col_list = list(qc_input_cells_threshold_duplicates_to_drop['orig_cols'])
            # drop duplicate columns from dataset
            print('dropping the following columns: ')
            print(drop_col_list)
            temp_input_cells = temp_input_cells.drop(columns=drop_col_list)
            # keep chosen ones by index, just for display purposes
            qc_input_cells_threshold_duplicates_to_keep = qc_input_cells_threshold_duplicates.loc[choose_duplicate_threshold_to_keep,:]
            print('updated after dropping duplicates: ')
            display(qc_input_cells_threshold_duplicates_to_keep)
        else:
            raise ValueError(
            f"choose_duplicate_threshold_to_keep must be a list. \n")
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # Update Main and ROI annotations areas with their respective exclusion areas
    # Subtract Exclusion areas from Main areas and from ROI areas
    print('Subtracting exclusion areas from their parent ROI and Main annotation areas: ')
    # cycle through all annot ids. If exclusion annot id is nonzero, the current annotation is an exclusion annotation. 
    # The main_annot_id and roi_annot_id columns for that row should show which parent annotations it belongs to. 
    # Need to go to those annotation rows and subtract the exclusion area from them. 
    for curr_annot_id in temp_input_area['full_annot_id']:
        print('curr_annot_id = ' + str(curr_annot_id))
        # if the current exclusion id is 0, it is not an exclusion. So we can skip this one. 
        if temp_input_area.loc[temp_input_area['full_annot_id']==curr_annot_id,'exclusion_annot_id'].iloc[0] == '0': continue
        
        # Now, the current exclusion id should be nonzero. 
        curr_area_to_subtract = temp_input_area.loc[temp_input_area['full_annot_id']==curr_annot_id,'Area in μm²'].iloc[0]
        curr_main_id = temp_input_area.loc[temp_input_area['full_annot_id']==curr_annot_id,'main_annot_id'].iloc[0]
        curr_roi_id = temp_input_area.loc[temp_input_area['full_annot_id']==curr_annot_id,'roi_annot_id'].iloc[0]

        # check
        print(f"Area to subtract: {curr_area_to_subtract:.2f}")
        print(f"Current Main ID: {curr_main_id}")
        print(f"Current ROI ID: {curr_roi_id}")
        
        # We are only operating on ROIs, so let's not worry about Mains (sometimes TBL doesn't export Main areas anymore)
        # Subtract exclusion area from its parent ROI
        temp_input_area.loc[((temp_input_area['main_annot_id']==curr_main_id) &
                            (temp_input_area['roi_annot_id']==curr_roi_id) &
                            (temp_input_area['exclusion_annot_id']=='0')),'Area in μm²'] -= curr_area_to_subtract
        # We are only operating on ROIs, so let's not worry about Mains (sometimes TBL doesn't export Main areas anymore)
        ## Subtract exclusion area from its parent Main
        #temp_input_area.loc[((temp_input_area['main_annot_id']==curr_main_id) &
        #                    (temp_input_area['roi_annot_id']=='0') &
        #                    (temp_input_area['exclusion_annot_id']=='0')),'Area in μm²'] -= curr_area_to_subtract

    # Now we can drop all exclusion annotations. (exclusion annot id equals zero)
    temp_input_area = temp_input_area.loc[temp_input_area['exclusion_annot_id']=='0']
    
    if show_meta:
        print('updated area: ')
        display(temp_input_area)
    
    # split each areas and cells df apart for each ROI annotation and store in a dict of dataframes. 
    # for cells, will be in 'parent_id' column. 
    # for areas, will be in 'full_annot_id' column. 
    # Note: Main annotations will be in the areas dict, but will not be processed because cell parent_id will never match their full_annot_id. 
    # Therefore only ROIs will be processed. 
    grouped_cells = temp_input_cells.groupby('parent_id')
    grouped_areas = temp_input_area.groupby('full_annot_id')
    # store each annotation separately in a dict
    cells_by_annot_dict = {parent_id: group for parent_id, group in grouped_cells}
    areas_by_annot_dict = {full_annot_id: group for full_annot_id, group in grouped_areas}

    if show_meta:
        print('annotations in cells: ')
        print([parent_id for parent_id, group in grouped_cells])
        print('annotations in areas: ')
        print([full_annot_id for full_annot_id, group in grouped_areas])

    # ---------------------------------------------------------
    # add region areas (per annot)
    # cells = add_region_areas(cells, area, microns_per_pixel)
    for curr_annot, curr_cells in cells_by_annot_dict.items():
        print(curr_annot)
        cells_by_annot_dict[curr_annot] = add_region_areas(curr_cells, areas_by_annot_dict[curr_annot], microns_per_pixel)
    
    # initialize a meta dict with the same keys as cells_by_annot_dict, but empty dataframes as values for now. 
    meta_dict = {key: pd.DataFrame() for key in cells_by_annot_dict}
    # initialize an output cdf dict with the same keys as cells_by_annot_dict, but empty dataframes as values for now. 
    cdf_dict = {key: pd.DataFrame() for key in cells_by_annot_dict}
    
    # ---------------------------------------------------------
    # Run the code to convert to pythologist CellDataFrame for each annotation. 
    # ---------------------------------------------------------
    for curr_annot, curr_cells in cells_by_annot_dict.items():
        print(' ')
        print('- - - - - - - - - - - - - - - - - - - - - - - -')
        print('processing ' + str(curr_annot) + ' - - - - - - -')
        # remove columns with only nans... again
        curr_cells = curr_cells.dropna(axis=1, how='all')
        # check
        #display(curr_cells.head(5))
        
        # Extract column metadata for the current annotation
        meta_dict[curr_annot] = extract_column_metadata(curr_cells)
        
        # rename markers list if specified
        if rename_markers_dict is not None:
            for old_marker_name, new_marker_name in rename_markers_dict.items():
                meta_dict[curr_annot] = rename_marker(meta_dict[curr_annot], old_marker_name, new_marker_name)
    
        if show_meta:
            print('meta for ' + str(curr_annot) + ':')
            display(meta_dict[curr_annot])
    
        # convert to pythologist CellDataFrame
        cdf = ingest_Lunaphore(curr_cells, 
                               meta_dict[curr_annot], 
                               proj_name=project_name, 
                               default_phenotype=default_phenotype, 
                               microns_per_pixel=microns_per_pixel,
                               overwrite_sample_name = overwrite_sample_name)

        # add cdf to the cdf_dict
        cdf_dict[curr_annot] = cdf

    # Now we have processed all the ROIs within the same sample separately, stored in the cdf_dict. 
    # Let's concatenate them all together in the same cdf to save, run qc, and/or return. 
    cdf = pd.concat(cdf_dict.values(), ignore_index=True)
    # They currently have separate sample_name and sample_id. Let's set them all the same. 
    # Same with project_id. 
    cdf['project_id'] = uuid.uuid4().hex
    cdf['sample_id'] = uuid.uuid4().hex
    if overwrite_sample_name is not None:
        cdf['sample_name'] = overwrite_sample_name
    else:
        # TODO: is this line broken?
        # cdf['sample_name'] = cdf['Annotation Group'].split('/')[1]
        # cdf['sample_name'] = cdf['Annotation Group'].iloc[0].split('/')[1]
        cdf['sample_name'] = cdf['Annotation Group'].str.split('/').str[1]
    # Combine sample_name with Parent Annotation (current frame_name) to get the updated frame_name
    cdf['frame_name'] = cdf['sample_name'] + '_' + cdf['frame_name']
    # set microns_per_pixel for combined cdf
    cdf.microns_per_pixel = microns_per_pixel
    
    if save_cdf:
        # save cdf, as .cdf.h5
        # example savefile_path: 'Processing/Nick_Horizon_Testing_20241219.cdf.h5'
        # savefile_dir
        # savefile_name
        savefile_path = os.path.join(savefile_dir, savefile_name + '.cdf.h5')


        # Troubleshooting (below):~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TODO: remove troubleshooting
        print("Microns per pixel stored values and type:")
        print(cdf.microns_per_pixel)
        print(type(cdf.microns_per_pixel))
        #
        #
        serialized_df = pd.DataFrame(cdf.serialize())
        serialized_df.to_hdf(savefile_path, 'data', mode='w', format='table', complib='zlib', complevel=9)
        print("See after pd.HDFStore:")
        with pd.HDFStore(savefile_path, mode='r') as store:
            print(store.keys())

        import h5py
        print("See after in h5py.File:")
        with h5py.File(savefile_path, 'r') as f:
            print(list(f.keys()))
        #
        #
        serialized_df = pd.DataFrame(cdf.serialize())
        test_path = savefile_path + ".test.h5"
        
        if os.path.exists(test_path):
            os.remove(test_path)
        
        serialized_df.to_hdf(test_path, 'data', mode='w', format='table', complib='zlib', complevel=9)
        print("See after pd.HDF.Store2:")
        with pd.HDFStore(test_path, mode='r') as store:
            print(store.keys())
        print("See after in h5py2.File:")
        import h5py
        with h5py.File(test_path, 'r') as f:
            print(list(f.keys()))
        # Troubleshooting (above ):~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        cdf.to_hdf(savefile_path,'data', mode='w')

    if run_qc:
        qc = cdf.qc()
        qc.run_tests()
        qc.print_results()
    
    # ----------------------------------
    # Process ROI measures/stats
    # ----------------------------------
    # cycle through all rois in the cdf and extract measures for them. 
    # first, initialize a list of dataframes to store roi measures in.
    roi_measures_list_wide = []
    roi_measures_list_long = []
    for curr_roi in cdf['Parent Annotation'].unique():
        cdf_sub = cdf.loc[cdf['Parent Annotation']==curr_roi]
        # get meta for current roi from the meta_dict created above
        curr_meta = meta_dict[curr_roi]
        # extract measures for current roi
        # outputs: roi_measures_wideform, roi_measures_longform
        curr_roi_measures_wide, curr_roi_measures_long = extract_roi_measures(cdf_sub, curr_meta, microns_per_pixel=microns_per_pixel)
        # add roi_measures to the list
        roi_measures_list_wide.append(curr_roi_measures_wide)
        roi_measures_list_long.append(curr_roi_measures_long)
    # Concatenate all roi_measures for this sample together
    all_roi_measures_wide = pd.concat(roi_measures_list_wide)
    all_roi_measures_long = pd.concat(roi_measures_list_long)
    # save measures
    if save_cdf:
        # save as csv file
        savefile_path_roi_measures_wide = os.path.join(savefile_dir, savefile_name + '_roi_measures_wide.csv')
        savefile_path_roi_measures_long = os.path.join(savefile_dir, savefile_name + '_roi_measures_long.csv')
        all_roi_measures_wide.to_csv(savefile_path_roi_measures_wide, index=False)
        all_roi_measures_long.to_csv(savefile_path_roi_measures_long, index=False)

    # Export Full Single Cell Data if requested
    if save_full_single_cell:
        export_comprehensive_single_cell(temp_input_cells, cdf, savefile_dir, savefile_name,
                                         microns_per_pixel=microns_per_pixel)

    if return_cdf:
        return cdf


# function to validate input parameters before proceeding
def validate_parameters(horizon_export_filepath, project_name, savefile_dir, savefile_name, microns_per_pixel, run_qc, return_cdf, save_full_single_cell, overwrite_sample_name, default_phenotype):
    # Type checking
    if not isinstance(horizon_export_filepath, str):
        raise TypeError("horizon_export_filepath must be a string")
    if not isinstance(project_name, str):
        raise TypeError("project_name must be a string")
    if not isinstance(savefile_dir, str):
        raise TypeError("savefile_dir must be a string")
    if not isinstance(savefile_name, str):
        raise TypeError("savefile_name must be a string")
    if not isinstance(microns_per_pixel, float):
        raise TypeError("microns_per_pixel must be a float")
    if not isinstance(run_qc, bool):
        raise TypeError("run_qc must be a boolean")
    if not isinstance(return_cdf, bool):
        raise TypeError("return_cdf must be a boolean")
    if not isinstance(save_full_single_cell, bool):
        raise TypeError("save_full_single_cell must be a boolean")
    if overwrite_sample_name is not None:
        if not isinstance(overwrite_sample_name, str):
                raise TypeError("overwrite_sample_name must be a string")
    if not isinstance(default_phenotype, str):
        raise TypeError("default_phenotype must be a string")
    # validate files exist
    if not os.path.isfile(horizon_export_filepath):
        raise ValueError(f"The path '{horizon_export_filepath}' is not a file")

# function to read in and process a horizon export file
# Will remove excluded cells, and separate areas and cells df. 
# Will annotate the annotations according to the nested structure we require. 
# Main annotation: 1.0.0
# ROI annotation: 1.1.0
# Exclusion annotation: 1.1.1
def import_horizon_file(horizon_export_filepath):
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

    return temp_input_cells, temp_input_area

# function to integrate region areas into the cells dataframe. 
# regions will just be called ANY (assuming there's only one)
def add_region_areas(cells_df, area_df, microns_per_pixel):
    # Set region label to ANY
    area_df['region_label'] = 'ANY'
    cells_df['region_label'] = 'ANY'
    # convert areas from microns into pixels, which is expected by pythologst. 
    # area_pixles2 = area_microns2*((1/microns_per_pixel)**2)
    area_df['Area_pixels_squared'] = area_df['Area in μm²'].apply(
        lambda x: round(microns2_to_pixels2(x, microns_per_pixel), 3))
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
        'X Position in μm':'x',
        'Y Position in μm':'y',
        'Area in μm²':'cell_area',
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
        df.loc[df['orig_cols'].str.contains(k, regex=False, na=False), 'Measurement_Type'] = v
    
    # ----------------------------
    # Extract Compartment Type
    df['Compartment_Type'] = ''
    for k,v in compartment_type_map.items():
        df.loc[df['orig_cols'].str.contains(k, regex=False, na=False), 'Compartment_Type'] = v
    
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
    #df['Marker'] = np.where((df['Compartment_Type']=='Nucleus') & (df['Measurement_Type']=='Threshold'), 
    df['Marker'] = np.where((df['Compartment_Type']=='Nucleus'), 
                            temp_marker_df2_nucleus.iloc[:, 0], 
                            temp_marker_df2.iloc[:, 0])
    # manually edit DAPI
    df.loc[df['orig_cols'].str.contains('DAPI', regex=False, na=False),'Marker'] = 'DAPI'
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
    
    # ----------------------------
    # Extract Channel Label
    # We must replicate the same logic for Channels to ensure they align
    
    # Cell/Cytoplasm Channels
    if temp_marker_df2.shape[1] > 1:
        temp_marker_df3 = temp_marker_df2.iloc[:, 1].str.split(')', expand=True)
    else:
        temp_marker_df3 = pd.DataFrame(index=df.index, columns=[0]) # Empty fallback

    # Nucleus Channels
    if temp_marker_df2_nucleus.shape[1] > 1:
        temp_marker_df3_nucleus = temp_marker_df2_nucleus.iloc[:, 1].str.split(')', expand=True)
    else:
        temp_marker_df3_nucleus = pd.DataFrame(index=df.index, columns=[0]) # Empty fallback
    
    # Apply logic
    #df['Channel'] = np.where((df['Compartment_Type']=='Nucleus') & (df['Measurement_Type']=='Threshold'),
    df['Channel'] = np.where(df['Compartment_Type']=='Nucleus',
                             temp_marker_df3_nucleus.iloc[:, 0],
                             temp_marker_df3.iloc[:, 0])

    # manually edit DAPI
    df.loc[df['orig_cols'].str.contains('DAPI', regex=False, na=False),'Channel'] = 'DAPI'
    
    # ----------------------------
    # Extract Thresholds
    # same for nuclei and other cell compartments
    temp_thresholds_df = df['orig_cols'].str.split('Threshold \\(', expand=True)
    
    if temp_thresholds_df.shape[1] > 1:
        # Take 2nd column, after the "Threshold ("
        temp_thresholds_df2 = temp_thresholds_df.iloc[:, 1].str.split(')', expand=True)
        # Take 1st column, before the ")"
        df['Threshold'] = temp_thresholds_df2.iloc[:, 0]
    else:
        df['Threshold'] = np.nan

    # ----------------------------
    # Edge cases:
    # manually set the Measurement Type for Area Threshold and Position Thresholds
    df.loc[(df['orig_cols'].str.contains('Area')) & (df['orig_cols'].str.contains('Threshold')),'Measurement_Type'] = 'Area Threshold'
    df.loc[(df['orig_cols'].str.contains('Position')) & (df['orig_cols'].str.contains('Threshold')),'Measurement_Type'] = 'Position Threshold'
    # By default let's use x and y positions of Cell (should be same as nuclei anyway)
    # so let's set Measurement_Type of the nucelei x and y to something specific...
    df.loc[(df['orig_cols'].str.contains('X Position in μm')) & (df['orig_cols'].str.contains('Nuclei')),'Measurement_Type'] = 'nucleus_x'
    df.loc[(df['orig_cols'].str.contains('Y Position in μm')) & (df['orig_cols'].str.contains('Nuclei')),'Measurement_Type'] = 'nucleus_y'
    # manually set Marker name to nan for these
    df.loc[(df['orig_cols'].str.contains('Area in μm²')) & (df['orig_cols'].str.contains('Threshold')),'Marker'] = np.nan
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


# function to transform Lunaphore Horizon output data into a pythologist CellDataFrame
def ingest_Lunaphore(df, 
                     meta, 
                     proj_name, 
                     default_phenotype='CD3', 
                     microns_per_pixel=0.28,
                     overwrite_sample_name = None):
    import numpy as np
    import uuid
    
    # drop Area Threshold and Position Threshold
    meta = meta.loc[~meta['Measurement_Type'].isin(['Area Threshold','Position Threshold'])]
    # drop DAPI
    meta = meta.loc[meta['Marker']!='DAPI']
    # drop nucleus_x and nucleus_y
    meta = meta.loc[~meta['Measurement_Type'].isin(['nucleus_x','nucleus_y'])]
    # drop Nucleus: Mean Intensity combination
    meta = meta.loc[~((meta['Compartment_Type']=='Nucleus') & (meta['Measurement_Type']=='Mean Intensity'))]
    
    # --------------------------------------------------------------------
    # Process Values
    # subset meta to name mappings for Cell (values)
    value_name_mappings = meta.loc[(meta['Compartment_Type'].isin(['Cell', 'Annotation'])) & (meta['Measurement_Type']!='Threshold')]
    
    # make column name remapping dict
    vals_remapping_dict = dict(zip(value_name_mappings['orig_cols'], value_name_mappings['Label_Mapping']))
    # get list of columns to keep 
    vals_cols_keep = list(value_name_mappings['orig_cols'])
    # subset data to include only values columns
    df_vals = df[vals_cols_keep].copy()
    print('values columns subset')
    df_vals = df_vals.rename(columns = vals_remapping_dict)
    print('values columns renamed')
    print(list(df_vals.columns))

    # ---------------------------------------------------------
    # Convert spatial measurements from microns to pixels
    # TODO: move this later
    # x, y are in microns -> pixels
    # cell_area is in μm² -> pixels²
    if 'x' in df_vals.columns:
        df_vals['x'] = df_vals['x'].apply(lambda v: microns_to_pixels(v, microns_per_pixel) if pd.notna(v) else v)

    if 'y' in df_vals.columns:
        df_vals['y'] = df_vals['y'].apply(lambda v: microns_to_pixels(v, microns_per_pixel) if pd.notna(v) else v)

    if 'cell_area' in df_vals.columns:
        df_vals['cell_area'] = df_vals['cell_area'].apply(lambda v: microns2_to_pixels2(v, microns_per_pixel) if pd.notna(v) else v)
    
    # TODO: makes sure this is applied correctly for single cell data. 
    # TODO: do we want to do nuclear area too?
    if 'nucleus_x' in df_vals.columns:
        df_vals['nucleus_x'] = df_vals['nucleus_x'].apply(lambda v: microns_to_pixels(v, microns_per_pixel) if pd.notna(v) else v)

    if 'nucleus_y' in df_vals.columns:
        df_vals['nucleus_y'] = df_vals['nucleus_y'].apply(lambda v: microns_to_pixels(v, microns_per_pixel) if pd.notna(v) else v)
    # ---------------------------------------------------------   
    
    # TODO: is there a better way to do this indexing? # TODO: ask Robert
    # Use these columns for index
    if 'Leiden clusters' in df_vals.columns:
        index_columns = ['Annotation Group','Parent Annotation','cell_index','cell_area','x','y','region_label','Leiden clusters']
    else:
        index_columns = ['Annotation Group','Parent Annotation','cell_index','cell_area','x','y','region_label']
    # Pull out regions column since it's not hashable (can't set it to index)
    # make the regions column into strings instead of dicts... (not ideal)
    #df_vals['regions'] = df_vals['regions'].apply(lambda x: str(x))
    df_regions = df_vals[index_columns + ['regions']].set_index(index_columns)
    # drop regions column from df_vals because it's not hashable. Will merge it back in later. 
    df_vals = df_vals.drop(columns=['regions'])
    
    # Want to collapse columns into a dictionary column. 
    # Set non value columns to index. 
    df_vals = df_vals.set_index(index_columns)
    # round to four decimal places. 
    df_vals = df_vals.round(4)
    # make channel values column and drop the individual values columns
    leftover_cols = list(df_vals.columns)
    df_vals['channel_values'] = df_vals.apply(lambda row: row.to_dict(), axis=1)
    df_vals = df_vals.drop(columns=leftover_cols)

    # Troubleshooting print statments: TODO: remove
    print("df_vals shape before calls merge:", df_vals.shape)
    print("df_vals index names:", df_vals.index.names)
    print("df_vals head:")
    print(df_vals.head(2))
    
    # --------------------------------------------------------------------
    # Process Thresholds
    # subset meta to name mappings for Threshold (calls)
    # Check for markers with multiple threshold compartments
    threshold_meta = meta.loc[meta['Measurement_Type'] == 'Threshold']
    marker_compartments = threshold_meta.groupby('Marker')['Compartment_Type'].nunique()
    multi_compartment_markers = marker_compartments[marker_compartments > 1]
    if not multi_compartment_markers.empty:
        warnings.warn(
            f"---------------------------------------------------------\n"
            f"The following markers have thresholds in multiple compartments: {list(multi_compartment_markers.index)}\n"
            f"This may cause unexpected behavior. Only one threshold per marker is typically used.\n"
            f"---------------------------------------------------------"
        )
        offending_markers_df = threshold_meta[threshold_meta['Marker'].isin(multi_compartment_markers.index)][['Marker', 'Compartment_Type', 'orig_cols']].sort_values('Marker')
        warnings.warn(f"Offending markers and their compartments:\n{offending_markers_df.to_string()}")

    call_name_mappings = meta.loc[(meta['Compartment_Type'].isin(['Cell', 'Nucleus', 'Cytoplasm', 'Annotation'])) & (meta['Measurement_Type']!='Mean Intensity')]
    # don't want to keep nucleus or cytoplasm areas...
    call_name_mappings = call_name_mappings.loc[~((call_name_mappings['Compartment_Type'].isin(['Nucleus','Cytoplasm'])) & (call_name_mappings['Measurement_Type']=='cell_area'))]
    
    # make column name remapping dict
    calls_remapping_dict = dict(zip(call_name_mappings['orig_cols'], call_name_mappings['Label_Mapping']))
    # get list of columns to keep 
    calls_cols_keep = list(call_name_mappings['orig_cols'])
    # subset data to include only threshold columns
    df_calls = df[calls_cols_keep].copy()
    print('calls columns subset')
    df_calls = df_calls.rename(columns = calls_remapping_dict)
    print('calls columns renamed')

    # ---------------------------------------------------------
    # Add same spatial conversion to calls as we did for values so we can merged
    # Convert spatial measurements from microns to pixels
    # so they match df_vals before setting index
    # TODO: do this and the df_vals later after we merge calls and vals back together, so we only have to do it once?
    if 'x' in df_calls.columns:
        df_calls['x'] = df_calls['x'].apply(
            lambda v: microns_to_pixels(v, microns_per_pixel) if pd.notna(v) else v
        )

    if 'y' in df_calls.columns:
        df_calls['y'] = df_calls['y'].apply(
            lambda v: microns_to_pixels(v, microns_per_pixel) if pd.notna(v) else v
        )

    if 'cell_area' in df_calls.columns:
        df_calls['cell_area'] = df_calls['cell_area'].apply(
            lambda v: microns2_to_pixels2(v, microns_per_pixel) if pd.notna(v) else v
        )
# ---------------------------------------------------------

    # drop regions column from df_calls because it's not hashable. Will merge it back in later. 
    df_calls = df_calls.drop(columns=['regions'])
    # Want to collapse columns into a dictionary column. 
    # Set non call columns to index. 
    df_calls = df_calls.set_index(index_columns)
    # pull out phenotype call
    if default_phenotype in list(df_calls.columns):
        # If the default phenotype is present, use that
        df_phenos = df_calls[[default_phenotype]]
    else:
        # if not, just use first scored call column
        # update default_phenotype to the first column
        default_phenotype = list(df_calls.columns)[0]
        df_phenos = df_calls[[default_phenotype]]
    # Drop that phenotype from the scored calls
    df_calls = df_calls.drop(columns=[default_phenotype])
    # Generate OTHER phenotype as the opposite boolean of the default phenotype
    df_phenos['OTHER'] = ~df_phenos[default_phenotype].astype(bool)
    # convert Booleans to 0 and 1s. For both scored calls and new phenotype
    df_calls = df_calls.astype(int)
    df_phenos = df_phenos.astype(int)
    # make scored calls column and drop the individual calls columns
    leftover_cols = list(df_calls.columns)
    df_calls['scored_calls'] = df_calls.apply(lambda row: row.to_dict(), axis=1)
    df_calls = df_calls.drop(columns=leftover_cols)
    # make phenotype calls column and drop the individual calls columns
    leftover_cols = list(df_phenos.columns)
    df_phenos['phenotype_calls'] = df_phenos.apply(lambda row: row.to_dict(), axis=1)
    # before dropping other calls, use the default phenotype column to create phenotype_label column
    df_phenos['phenotype_label'] = np.where(df_phenos[default_phenotype] == 1, default_phenotype, 'OTHER')
    # drop other calls columns
    df_phenos = df_phenos.drop(columns=leftover_cols)
    
    # merge calls and phenotypes
    df_phenos_calls = df_calls.merge(df_phenos, how='inner', left_index=True, right_index=True)

    # Troubleshooting print statments: TODO: remove
    print("df_calls shape before phenos merge:", df_calls.shape)
    print("df_phenos shape:", df_phenos.shape)
    print("df_phenos_calls shape:", df_phenos_calls.shape)
    print("df_phenos_calls index names:", df_phenos_calls.index.names)
    print("df_phenos_calls head:")
    print(df_phenos_calls.head(2))

    # --------------------------------------------------------------------
    # Merge vals and calls back together on the indexes. 
    df_merge = df_vals.merge(df_phenos_calls, how='inner', left_index=True, right_index=True)

    # troubleshooting print statments: TODO: remove
    print("df_merge shape after vals/calls merge:", df_merge.shape)
    print("df_merge head after vals/calls merge:")
    print(df_merge.head(2))

    # merge the regions column back in using indexes. 
    df_merge = df_merge.merge(df_regions, how='inner', left_index=True, right_index=True)

    # troubleshooting print statments: TODO: remove
    print("df_regions shape:", df_regions.shape)
    print("df_merge shape after regions merge:", df_merge.shape)
    print("df_merge head after regions merge:")
    print(df_merge.head(2))
    
    # Add some other pythologist columns
    # neighbors	frame_name	frame_id	sample_name	project_name	sample_id	project_id	frame_shape
    df_merge = df_merge.reset_index()
    df_merge['neighbors'] = np.nan
    df_merge['frame_id'] = uuid.uuid4().hex
    if overwrite_sample_name is not None:
        df_merge['sample_name'] = overwrite_sample_name
        #df_merge['frame_name'] = overwrite_sample_name
    else:
        df_merge['sample_name'] = df_merge['Parent Annotation']
    df_merge['frame_name'] = df_merge['Parent Annotation']
    df_merge['project_name'] = proj_name
    df_merge['project_id'] = uuid.uuid4().hex
    df_merge['sample_id'] = uuid.uuid4().hex
    # I should calculate this using the minimum and maximum cell centroid locations for each ROI...
    # TODO: do we want this in pixels or microns? Is this implemented correctly?
    df_merge['frame_shape'] = df_merge['region_label'].apply(lambda x: tuple([1000,1000]))
    # width_px = int(np.ceil(df_merge['x'].max())) + 1 if 'x' in df_merge.columns else 1000
    # height_px = int(np.ceil(df_merge['y'].max())) + 1 if 'y' in df_merge.columns else 1000
    # df_merge['frame_shape'] = [(height_px, width_px)] * len(df_merge)
    
    # convert to pythologist CellDataFrame
    cdf = CellDataFrame(df_merge)
    # 0.28 microns/pixel
    cdf.microns_per_pixel = microns_per_pixel
    
    return cdf

# Input: pythologist cdf that has been ingested from Lunaphore
# Calculate QC measures per ROI. 
# Fluorescence per marker: Mean, Median, Min, Max, Standard Deviation
# Cell Area: Mean, Median, Min, Max, Standard Deviation
# Cell density per phenotype in this ROI (including all cells)
def extract_roi_measures(cdf, meta, microns_per_pixel=0.28):

    # function to calculate gini index
    def gini_coefficient(x):
        # Remove NaN values
        x = x[~np.isnan(x)]
        # Handle edge cases
        if len(x) == 0:
            return np.nan
        if len(x) == 1 or np.all(x == x[0]):
            return 0.0
        # Sort the values
        x_sorted = np.sort(x)
        n = len(x_sorted)
        # Calculate Gini coefficient
        cumsum = np.cumsum(x_sorted)
        gini = (2 * np.sum((np.arange(1, n + 1) * x_sorted))) / (n * cumsum[-1]) - (n + 1) / n
        return gini

    # Cell Areas, in millimeters squared
    cell_area_um2 = cdf['cell_area'] * (microns_per_pixel ** 2)
    cell_area_mm2 = cell_area_um2 * 0.000001
    #
    cell_area_mean = cell_area_mm2.mean()
    cell_area_median = cell_area_mm2.median()
    cell_area_min = cell_area_mm2.min()
    cell_area_max = cell_area_mm2.max()
    cell_area_std = cell_area_mm2.std()
    cell_area_skew = cell_area_mm2.skew()
    cell_area_kurtosis = cell_area_mm2.kurtosis()
    cell_area_gini = gini_coefficient(cell_area_mm2.values)

    # Imports
    import ast

    # Fluorescence Measurements
    # Ensure they're dicts and not strings
    cdf['channel_values'] = cdf['channel_values'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # expand dict column. New df will have keys as columns. 
    fluorescence_df_pre = pd.json_normalize(cdf['channel_values'])
    # get fluorescence stats, adding cell_area_mm2 values as a column to the end.
    fluorescence_mean_df = pd.DataFrame([fluorescence_df_pre.mean()]).assign(cell_area_mm2=cell_area_mean)
    fluorescence_median_df = pd.DataFrame([fluorescence_df_pre.median()]).assign(cell_area_mm2=cell_area_median)
    fluorescence_min_df = pd.DataFrame([fluorescence_df_pre.min()]).assign(cell_area_mm2=cell_area_min)
    fluorescence_max_df = pd.DataFrame([fluorescence_df_pre.max()]).assign(cell_area_mm2=cell_area_max)
    fluorescence_std_df = pd.DataFrame([fluorescence_df_pre.std()]).assign(cell_area_mm2=cell_area_std)
    fluorescence_skew_df = pd.DataFrame([fluorescence_df_pre.skew()]).assign(cell_area_mm2=cell_area_skew)
    fluorescence_kurtosis_df = pd.DataFrame([fluorescence_df_pre.kurtosis()]).assign(cell_area_mm2=cell_area_kurtosis)
    fluorescence_gini_df = pd.DataFrame([fluorescence_df_pre.apply(lambda col: gini_coefficient(col.values))]).assign(cell_area_mm2=cell_area_gini)

    # Phenotype Measurements
    # Ensure they're dicts and not strings
    cdf['phenotype_calls'] = cdf['phenotype_calls'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # expand dict column. New df will have keys as columns. 
    phenotype_df = pd.json_normalize(cdf['phenotype_calls'])
    if 'OTHER' in phenotype_df.columns:
        phenotype_df_pre = phenotype_df.drop('OTHER', axis=1)

    # Scored Call Measurements
    # Ensure they're dicts and not strings
    cdf['scored_calls'] = cdf['scored_calls'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # expand dict column. New df will have keys as columns. 
    scored_calls_df = pd.json_normalize(cdf['scored_calls'])

    # Combine phenotypes and scored calls
    phenotype_df = pd.concat([phenotype_df_pre,scored_calls_df], axis=1)

    # Calculate the number of positive cells for each phenotype, "count"
    counts_df = pd.DataFrame(phenotype_df.sum()).T

    # Calculate the cell densities by dividing by the area
    # Ensure they're dicts and not strings
    cdf['regions'] = cdf['regions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # expand dict column. New df will have keys as columns. 
    curr_roi_area_df = pd.json_normalize(cdf['regions'])
    curr_roi_area_pixels2 = curr_roi_area_df['ANY'].iloc[0]
    curr_roi_area_mm2 = curr_roi_area_pixels2 * (microns_per_pixel ** 2) * 0.000001
    densities_df_pixels2 = counts_df / curr_roi_area_pixels2
    densities_df_mm2 = counts_df / curr_roi_area_mm2
    # combine
    roi_areas_df = pd.DataFrame({'roi_area_pixels2':curr_roi_area_pixels2,
                            'roi_area_mm2':curr_roi_area_mm2}, 
                             index=[0])
    roi_areas_df = roi_areas_df.reset_index(drop=True)

    # Get sample and roi labels
    labs_keep = ['Annotation Group','Parent Annotation','sample_name','frame_name','project_name']
    labs_cols = pd.DataFrame(cdf[labs_keep]).head(1).reset_index(drop=True)
    # add ingestion date
    labs_cols['ingestion_date'] = pd.Timestamp.today().strftime('%Y-%m-%d')

    # get thresholds to add in
    thresholds_df = get_thresholds_from_meta(meta)
    # transform thresholds_df for wideform 
    thresholds_df_wideform = thresholds_df.copy()
    # first, add compartment to marker name
    for col in thresholds_df_wideform.columns:
        if col == 'data_type': continue
        thresholds_df_wideform = thresholds_df_wideform.rename(columns={col:col + '_' + str(thresholds_df_wideform.loc[thresholds_df_wideform['data_type']=='Compartment_Type',col].values[0])})
    # take only Threshold values for wideform
    thresholds_df_wideform = thresholds_df_wideform.loc[thresholds_df_wideform['data_type']=='Threshold']
    thresholds_df_wideform = thresholds_df_wideform.drop(columns=['data_type']).add_suffix('_Threshold').reset_index(drop=True)

    # clean up indexes if needed
    labs_cols = labs_cols.reset_index(drop=True)
    roi_areas_df = roi_areas_df.reset_index(drop=True)

    # Aggregate all measures together into one datafrome
    # note fluorescence dfs have corresponding cell_area_mm2 stats in them. 
    roi_measures_wideform = pd.concat([labs_cols,
                             roi_areas_df,
                             counts_df.add_suffix('_counts'),
                             densities_df_pixels2.add_suffix('_density_pixels2'),
                             densities_df_mm2.add_suffix('_density_mm2'),
                             fluorescence_mean_df.add_suffix('_mean'),
                             fluorescence_median_df.add_suffix('_median'),
                             fluorescence_min_df.add_suffix('_min'),
                             fluorescence_max_df.add_suffix('_max'),
                             fluorescence_std_df.add_suffix('_std'),
                             fluorescence_skew_df.add_suffix('_skew'),
                             fluorescence_kurtosis_df.add_suffix('_kurtosis'),
                             fluorescence_gini_df.add_suffix('_gini'),
                             thresholds_df_wideform
                             ],
                             axis=1)

    # Create a long-form version
    roi_measures_longform_rois = pd.concat([labs_cols, 
                                            roi_areas_df
                                            ],
                                            axis=1)

    roi_measures_longform_cells = pd.concat([
                             thresholds_df,
                             counts_df.assign(data_type='count'),
                             densities_df_pixels2.assign(data_type='density_pixels2'),
                             densities_df_mm2.assign(data_type='density_mm2'),
                             fluorescence_mean_df.assign(data_type='mean'),
                             fluorescence_median_df.assign(data_type='median'),
                             fluorescence_min_df.assign(data_type='min'),
                             fluorescence_max_df.assign(data_type='max'),
                             fluorescence_std_df.assign(data_type='std'),
                             fluorescence_skew_df.assign(data_type='skew'),
                             fluorescence_kurtosis_df.assign(data_type='kurtosis'),
                             fluorescence_gini_df.assign(data_type='gini')
                             ],
                             axis=0)
    
    # repeat the roi info to match the number of rows before concatenating. 
    roi_measures_longform_rois_matchnumrows = pd.DataFrame(np.repeat(roi_measures_longform_rois.values, len(roi_measures_longform_cells.index), axis=0), 
                                   columns=roi_measures_longform_rois.columns)
    
    # concat them together
    roi_measures_longform = pd.concat([roi_measures_longform_rois_matchnumrows.reset_index(drop=True),
                                       roi_measures_longform_cells.reset_index(drop=True)],
                                       axis=1)
    
    return roi_measures_wideform, roi_measures_longform


def get_thresholds_from_meta(meta):
    thresholds_df_pre = meta.loc[meta['Measurement_Type']=='Threshold']
    # subset to columns of interest, set Marker as index, Transpose to get Markers as col names
    thresholds_df = thresholds_df_pre[['Compartment_Type','Marker','Channel','Threshold']].set_index('Marker').T.reset_index().rename(columns={'index': 'data_type'})
    # remove name from columns list
    thresholds_df.columns.name = None
    
    return thresholds_df


# =========================================================================
# NEW HELPER FUNCTION: Create Consistent Column Names
# =========================================================================
def _create_consistent_column_name(row):
    """
    Creates a new, consistent column name from a metadata row
    generated by extract_column_metadata.
    """
    orig_col = row['orig_cols']
    
    # Columns to keep as is
    identity_cols = [
        'Annotation Group', 'cell_index', 'Region', 'Class group', 'annot_name',
        'Parent Annotation', 'parent_main_id', 'parent_roi_id', 'sample_name',
        'frame_name', 'project_name', 'sample_id', 'project_id', 'frame_id',
        'Leiden clusters'
    ]
    if orig_col in identity_cols:
        return orig_col

    compartment = row['Compartment_Type']
    measurement = row['Measurement_Type']
    marker = row['Marker']

    # Standardize compartment names
    if compartment == 'Nuclei': compartment = 'Nucleus'
    elif compartment == 'Cells': compartment = 'Cell'
    elif compartment == 'Cytoplasm': compartment = 'Cytoplasm'

    # ---------------------------------------------------------
    # Minimal special handling so exported single-cell names
    # match what export_comprehensive_single_cell() expects
    # ---------------------------------------------------------
    if measurement == 'x':
        if compartment == 'Nucleus':
            return 'nucleus_x'
        elif compartment == 'Cytoplasm':
            return 'cytoplasm_x'
        else:
            return 'x_um_raw'
    
    if measurement == 'y':
        if compartment == 'Nucleus':
            return 'nucleus_y'
        elif compartment == 'Cytoplasm':
            return 'cytoplasm_y'
        else:
            return 'y_um_raw'
    
    if measurement == 'cell_area':
        if compartment == 'Nucleus':
            return 'nucleus_area'
        elif compartment == 'Cytoplasm':
            return 'cytoplasm_area'
        else:
            return 'cell_area_um2_raw'
        

    if measurement == 'nucleus_x':
        return 'nucleus_x'
    if measurement == 'nucleus_y':
        return 'nucleus_y'


    parts = []
    if pd.notna(compartment) and compartment != 'Annotation':
        parts.append(compartment)

    # Standardize measurement names
    measurement_map = {
        # 'cell_area': 'Area_um2', 'x': 'X_Position_Cell_um', 'y': 'Y_Position_Cell_um',
        'Mean Intensity': 'MeanIntensity', 
        'Threshold': 'Threshold',
        # 'nucleus_x': 'X_Position_Nucleus_um', 'nucleus_y': 'Y_Position_Nucleus_um',
        'Area Threshold': 'Area_Threshold', 
        'Position Threshold': 'Position_Threshold'
    }
    if measurement in measurement_map:
        parts.append(measurement_map[measurement])
    else:
        return orig_col # Return original if no rule

    # Add marker if it exists, and clean it
    if pd.notna(marker) and marker != 'DAPI':
        marker = marker.replace(' ', '').replace('-', '')
        parts.append(marker)

    return '_'.join(parts)

# =========================================================================
# NEW FUNCTION: Export Comprehensive Single Cell Data
# =========================================================================
def export_comprehensive_single_cell(temp_input_cells, cdf, savefile_dir, savefile_name, microns_per_pixel=0.28):
    """
    Export a comprehensive single-cell table that:
    1) merges processed CDF metadata back onto raw Horizon single-cell data
    2) renames columns to be more CDF-like
    3) includes x, y, and cell_area in both pixels and microns
    
    temp_input_cells: the raw single-cell dataframe output from Lunaphore Horizon, before any processing
    cdf: the processed CellDataFrame that has been ingested and processed in pythologist, which contains the final metadata and identifiers for each cell
    savefile_dir: directory to save the output CSV file
    savefile_name: name of the output CSV file (without extension)
    microns_per_pixel: scaling factor for converting between microns and pixels
    """
   
    # ------------------------------------------------------------------
    # 1. Pull useful processed CDF columns
    # ------------------------------------------------------------------
    cdf_meta_cols = [
        'cell_index',
        'Annotation Group',
        'Parent Annotation',
        'sample_name',
        'frame_name',
        'project_name',
        'sample_id',
        'project_id',
        'frame_id',
        'region_label',
        'regions',
        'frame_shape',
        'neighbors',
        'channel_values',
        'scored_calls',
        'phenotype_calls',
        'phenotype_label',
        'x',
        'y',
        'cell_area'
    ]
    cdf_meta_cols = [c for c in cdf_meta_cols if c in cdf.columns]

    cdf_meta = cdf[cdf_meta_cols].copy()

    merge_keys = ['cell_index', 'Annotation Group', 'Parent Annotation']
    cdf_meta = cdf_meta.drop_duplicates(subset=merge_keys)

    # ------------------------------------------------------------------
    # 2. Prepare raw single-cell table
    # ------------------------------------------------------------------
    temp_full = temp_input_cells.copy()
    temp_full = temp_full.rename(columns={
        'Annotation Index': 'cell_index',
        'parent_id': 'Parent Annotation'
    })

    # ------------------------------------------------------------------
    # 3. Merge processed CDF data onto raw full single-cell data
    # ------------------------------------------------------------------
    full_sc_data = pd.merge(
        cdf_meta,
        temp_full,
        on=merge_keys,
        how='inner'
    )

    # ------------------------------------------------------------------
    # 4. Rename raw Horizon columns into more CDF-like naming
    # ------------------------------------------------------------------
    sc_meta = extract_column_metadata(full_sc_data)

    rename_dict = {
        row['orig_cols']: _create_consistent_column_name(row)
        for _, row in sc_meta.iterrows()
    }

    full_sc_data = full_sc_data.rename(columns=rename_dict)

    # ------------------------------------------------------------------
    # 5. Add explicit pixel and micron versions of CDF spatial columns
    #    CDF x, y, cell_area are already in pixels / pixels²
    # ------------------------------------------------------------------
    if 'x' in full_sc_data.columns:
        full_sc_data['x_pixels'] = full_sc_data['x']
        full_sc_data['x_um'] = full_sc_data['x'].apply(
            lambda v: pixels_to_microns(v, microns_per_pixel) if pd.notna(v) else v
        )

    if 'y' in full_sc_data.columns:
        full_sc_data['y_pixels'] = full_sc_data['y']
        full_sc_data['y_um'] = full_sc_data['y'].apply(
            lambda v: pixels_to_microns(v, microns_per_pixel) if pd.notna(v) else v
        )

    if 'cell_area' in full_sc_data.columns:
        full_sc_data['cell_area_pixels2'] = full_sc_data['cell_area']
        full_sc_data['cell_area_um2'] = full_sc_data['cell_area'].apply(
            lambda v: pixels2_to_microns2(v, microns_per_pixel) if pd.notna(v) else v
        )
    
    # Also convert raw cell micron coordinates to pixels for comparison
    if 'x_um_raw' in full_sc_data.columns:
        full_sc_data['x_raw_pixels'] = full_sc_data['x_um_raw'].apply(
            lambda v: microns_to_pixels(v, microns_per_pixel) if pd.notna(v) else v
        )

    if 'y_um_raw' in full_sc_data.columns:
        full_sc_data['y_raw_pixels'] = full_sc_data['y_um_raw'].apply(
            lambda v: microns_to_pixels(v, microns_per_pixel) if pd.notna(v) else v
        )

    # ------------------------------------------------------------------
    # 6. If raw nucleus/cytoplasm columns exist in micron units after renaming,
    #    add pixel versions too
    # ------------------------------------------------------------------
    extra_length_um_cols = ['nucleus_x', 'nucleus_y', 'cytoplasm_x', 'cytoplasm_y']
    extra_area_um_cols = ['nucleus_area', 'cytoplasm_area']

    for col in extra_length_um_cols:
        if col in full_sc_data.columns:
            full_sc_data[col + '_pixels'] = full_sc_data[col].apply(
                lambda v: microns_to_pixels(v, microns_per_pixel) if pd.notna(v) else v
            )
            full_sc_data[col + '_um'] = full_sc_data[col]

    for col in extra_area_um_cols:
        if col in full_sc_data.columns:
            full_sc_data[col + '_pixels2'] = full_sc_data[col].apply(
                lambda v: microns2_to_pixels2(v, microns_per_pixel) if pd.notna(v) else v
            )
            full_sc_data[col + '_um2'] = full_sc_data[col]

    # ------------------------------------------------------------------
    # 7. Round numeric unit-converted columns
    # ------------------------------------------------------------------
    cols_to_round = [
        'x', 'y', 'cell_area',
        'x_pixels', 'y_pixels', 'cell_area_pixels2',
        'x_um', 'y_um', 'cell_area_um2',
        'x_raw_pixels', 'y_raw_pixels'
    ]
    cols_to_round += [c for c in full_sc_data.columns if c.endswith('_pixels')]
    cols_to_round += [c for c in full_sc_data.columns if c.endswith('_pixels2')]
    cols_to_round += [c for c in full_sc_data.columns if c.endswith('_um')]
    cols_to_round += [c for c in full_sc_data.columns if c.endswith('_um2')]

    cols_to_round = [c for c in cols_to_round if c in full_sc_data.columns]
    cols_to_round = list(dict.fromkeys(cols_to_round))  # preserve order, remove duplicates

    if cols_to_round:
        full_sc_data.loc[:, cols_to_round] = full_sc_data.loc[:, cols_to_round].round(4)

    dupe_cols = full_sc_data.columns[full_sc_data.columns.duplicated()].tolist()
    print("Duplicate column names in full_sc_data:", dupe_cols)

    from collections import Counter
    dupe_round = [k for k, v in Counter(cols_to_round).items() if v > 1]
    print("Duplicate names in cols_to_round:", dupe_round)

    # ------------------------------------------------------------------
    # 8. Reorder columns so processed/CDF-like fields come first
    # ------------------------------------------------------------------
    preferred_first = [
        'cell_index',
        'Annotation Group',
        'Parent Annotation',
        'sample_name',
        'frame_name',
        'project_name',
        'sample_id',
        'project_id',
        'frame_id',
        'x',
        'y',
        'cell_area',
        'x_pixels',
        'y_pixels',
        'cell_area_pixels2',
        'x_um',
        'y_um',
        'x_um_raw',
        'y_um_raw',
        'x_raw_pixels',
        'y_raw_pixels',
        'cell_area_um2',
        'region_label',
        'regions',
        'frame_shape',
        'neighbors',
        'phenotype_label',
        'phenotype_calls',
        'scored_calls',
        'channel_values'
    ]
    preferred_first = [c for c in preferred_first if c in full_sc_data.columns]

    remaining_cols = [c for c in full_sc_data.columns if c not in preferred_first]
    full_sc_data = full_sc_data[preferred_first + remaining_cols]

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    savefile_path = os.path.join(savefile_dir, savefile_name + '_full_single_cell.csv')
    full_sc_data.to_csv(savefile_path, index=False)
    print(f"Comprehensive single-cell data successfully saved to: {savefile_path}")

    return full_sc_data


def microns_to_pixels(value_um, microns_per_pixel):
    """
    Converts a length in microns to pixels using the provided microns per pixel scaling factor.
    value_um = length in microns
    microns_per_pixel = scaling factor (microns/pixel)
    returns length in pixels
    """
    return value_um / microns_per_pixel

def microns2_to_pixels2(value_um2, microns_per_pixel):
    """
    Converts an area in microns squared to pixels squared using the provided microns per pixel scaling factor.
    value_um2 = area in microns squared
    microns_per_pixel = scaling factor (microns/pixel)
    returns area in pixels squared
    """
    return value_um2 / (microns_per_pixel ** 2)

def pixels_to_microns(value_px, microns_per_pixel):
    """
    Converts a length in pixels to microns using the provided microns per pixel scaling factor.
    value_px = length in pixels
    microns_per_pixel = scaling factor (microns/pixel)
    returns length in microns
    """
    return value_px * microns_per_pixel

def pixels2_to_microns2(value_px2, microns_per_pixel):
    """
    Converts an area in pixels squared to microns squared using the provided microns per pixel scaling factor.
    value_px2 = area in pixels squared
    microns_per_pixel = scaling factor (microns/pixel)
    returns area in microns squared
    """
    return value_px2 * (microns_per_pixel ** 2)