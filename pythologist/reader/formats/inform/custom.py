from pythologist.reader.formats.inform.frame import CellFrameInForm
from pythologist.reader.formats.inform.sets import CellSampleInForm, CellProjectInForm
from pythologist.image_utilities import read_tiff_stack, make_binary_image_array, binary_image_dilation
from uuid import uuid4
import pandas as pd
import numpy as np
import os, re, sys

class CellProjectInFormCustomMask(CellProjectInForm):
    """
    Read in a project that has a region set by a custon hand drawn area

    Accessed via ``read_path`` with the additonal parameters
    """
    def read_path(self,*args,**kwargs):
        """
        Read in the project folder

        Args: 
            path (str): location of the project directory
            project_name (str): name of the project
            sample_name_index (int): where in the directory chain is the foldername that is the sample name if not set use full path.  -1 is last directory
            inform_analysis_dict (dict): dictionary of shortcuts to translate to simpler channel names
            channel_abbreviations (dict): dictionary of shortcuts to translate to simpler channel names
            verbose (bool): if true print extra details
            require (bool): if true (default), require that channel componenet image be present
            skip_segmentation_processing (bool): if false (default), it will store the cellmap and edgemap images, if true, it will skip these steps to save time but downstream applications will not be able to generate the cell-cell contact measurements or segmentation images.
            microns_per_pixel (float): conversion factor
            custom_mask_name (str): the mask name that will end in <maskname>.tif
            other_mask_name (str): what you want to call areas not contained in your custom mask
            alternate_annotation_path (str): if None (default) look for annotations with the inform files
        """
        super().read_path(*args,**kwargs)

    def create_cell_sample_class(self):
        return CellSampleInFormCustomMask()

class CellSampleInFormCustomMask(CellSampleInForm):
    def create_cell_frame_class(self):
        return CellFrameInFormCustomMask()
    def read_path(self,path,sample_name=None,
                            inform_analysis_dict=None,
                            channel_abbreviations=None,
                            verbose=False,require_component=True,
                            require_score=True,
                            skip_segmentation_processing=False,
                            custom_mask_name='Tumor',
                            other_mask_name='Stroma',
                            alternate_annotation_path=None,
                            dry_run=False
                            ):
        if sample_name is None: sample_name = path
        if not os.path.isdir(path):
            raise ValueError('Path input must be a directory')
        absdir = os.path.abspath(path)
        z = 0
        files = [x for x in os.listdir(path) if not x[0]=='.']
        z += 1
        segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
        if len(segs) == 0: raise ValueError("There needs to be cell_seg_data in the folder.")
        frames = []
        for file in segs:
            m = re.match('(.*)cell_seg_data.txt$',file)
            score = os.path.join(path,m.group(1)+'score_data.txt')
            if ((not os.path.exists(score)) and (not require_score)):
                # score is now none
                #print("set to none")
                score = None
            #summary = os.path.join(path,m.group(1)+'cell_seg_data_summary.txt')
            binary_seg_maps = os.path.join(path,m.group(1)+'binary_seg_maps.tif')
            component_image = os.path.join(path,m.group(1)+'component_data.tif')
            tfile = os.path.join(path,m.group(1)+'tissue_seg_data.txt')
            tumor = os.path.join(path,m.group(1)+custom_mask_name+'.tif') if alternate_annotation_path is None \
                else os.path.join(alternate_annotation_path,m.group(1)+custom_mask_name+'.tif')
            if not os.path.exists(tumor): raise ValueError("Custom mask is missing")
            tissue_seg_data = tfile if os.path.exists(tfile) else None
            frame = m.group(1).rstrip('_')
            data = os.path.join(path,file)
            if require_score and not os.path.exists(score):
                    raise ValueError('Missing score file '+score)
            if verbose: sys.stderr.write('Acquiring frame '+data+"\n")
            cid = self.create_cell_frame_class()
            cid.read_raw(frame_name = frame,
                         cell_seg_data_file=data,
                         score_data_file=score,
                         inform_analysis_dict=inform_analysis_dict,
                         #tissue_seg_data_file=tissue_seg_data,
                         binary_seg_image_file=binary_seg_maps,
                         component_image_file=component_image,
                         #channel_abbreviations=channel_abbreviations,
                         verbose=verbose,
                         require_component=require_component,
                         skip_segmentation_processing=skip_segmentation_processing,
                         require_score=require_score,
                         dry_run=dry_run

                         )
            if verbose: sys.stderr.write("setting mask and not mask\n")
            cid.set_area(tumor,
                         custom_mask_name,
                         other_mask_name,
                         verbose=verbose

                         )
            frame_id = cid.id
            self._frames[frame_id]=cid
            frames.append({'frame_id':frame_id,'frame_name':frame,'frame_path':absdir})
            if verbose: sys.stderr.write("finished mask and not mask\n")
        self._key = pd.DataFrame(frames)
        self._key.index.name = 'db_id'
        self.sample_name = sample_name
         
class CellFrameInFormCustomMask(CellFrameInForm):
    def __init__(self):
        super().__init__()
        #self.data_tables['custom_images'] = {'index':'db_id',
        #         'columns':['custom_label','image_id']}
        for x in self.data_tables.keys():
            if x in self._data: continue
            self._data[x] = pd.DataFrame(columns=self.data_tables[x]['columns'])
            self._data[x].index.name = self.data_tables[x]['index']
    def set_area(self,area_image,custom_mask_name,other_mask_name,verbose=False):
        area_binary = read_tiff_stack(area_image)[0]['raw_image']
        area_binary = make_binary_image_array(area_binary)
        image_id= uuid4().hex
        self._images[image_id] = area_binary
        df = pd.DataFrame({'mask_label':['Area'],'image_id':[image_id]})
        df.index.name = 'db_id'
        self.set_data('mask_images',df)

        processed_image = self.get_image(self.processed_image_id).astype(np.uint8)
        #margin_binary = grown&processed_image
        tumor_binary = area_binary&processed_image
        stroma_binary = (~(tumor_binary&processed_image))&processed_image
        d = {custom_mask_name:tumor_binary,
             other_mask_name:stroma_binary}
        self.set_regions('InFormCustomMask',d,description="Set from a single custom mask.")
        return d

class CellProjectInFormLineArea(CellProjectInForm):
    """
    Read in a project that has a region set by a custon hand drawn area, and a margin set by a line

    Accessed via ``read_path`` with the additonal parameters
    """
    def create_cell_sample_class(self):
        return CellSampleInFormLineArea()
    def read_path(self,*args,**kwargs):
        """
        Read in the project folder

        Args: 
            path (str): location of the project directory
            project_name (str): name of the project
            sample_name_index (int): where in the directory chain is the foldername that is the sample name if not set use full path.  -1 is last directory
            inform_analysis_dict (dict): dictionary of shortcuts to translate to simpler channel names
            verbose (bool): if true print extra details
            require (bool): if true (default), require that channel componenet image be present
            require_score (bool): if true (default), require that score be present
            skip_segmentation_processing (bool): if false (default), it will store the cellmap and edgemap images, if true, it will skip these steps to save time but downstream applications will not be able to generate the cell-cell contact measurements or segmentation images.
            microns_per_pixel (float): conversion factor
            steps (int): how many pixels out from the hand drawn line to consider the margin
            alternate_annotation_path (str): if None (default) look for annotations with the inform files
        """
        super().read_path(*args,**kwargs)

class CellSampleInFormLineArea(CellSampleInForm):
    def create_cell_frame_class(self):
        return CellFrameInFormLineArea()
    def read_path(self,path,sample_name=None,
                            inform_analysis_dict=None,
                            verbose=False,
                            require_component=True,
                            require_score=True,
                            skip_segmentation_processing=False,
                            steps=76,
                            alternate_annotation_path=None,
                            dry_run=False,
                            ):
        if sample_name is None: sample_name = path
        if not os.path.isdir(path):
            raise ValueError('Path input must be a directory')
        absdir = os.path.abspath(path)
        z = 0
        files = [x for x in os.listdir(path) if not x[0]=='.']
        z += 1
        segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
        if len(segs) == 0: raise ValueError("There needs to be cell_seg_data in the folder.")
        frames = []
        for file in segs:
            m = re.match('(.*)cell_seg_data.txt$',file)
            score = os.path.join(path,m.group(1)+'score_data.txt')
            if ((not os.path.exists(score)) and (not require_score)):
                # score is now none
                #print("set to none")
                score = None
            #summary = os.path.join(path,m.group(1)+'cell_seg_data_summary.txt')
            binary_seg_maps = os.path.join(path,m.group(1)+'binary_seg_maps.tif')
            component_image = os.path.join(path,m.group(1)+'component_data.tif')
            tfile = os.path.join(path,m.group(1)+'tissue_seg_data.txt')
            tumor = os.path.join(path,m.group(1)+'Tumor.tif') if alternate_annotation_path is None \
                else os.path.join(alternate_annotation_path,m.group(1)+'Tumor.tif')
            margin = os.path.join(path,m.group(1)+'Invasive_Margin.tif') if alternate_annotation_path is None \
                else os.path.join(alternate_annotation_path,m.group(1)+'Invasive_Margin.tif')
            tissue_seg_data = tfile if os.path.exists(tfile) else None
            frame = m.group(1).rstrip('_')
            data = os.path.join(path,file)
            if require_score and not os.path.exists(score):
                    raise ValueError('Missing score file '+score)
            if verbose: sys.stderr.write('Acquiring frame '+data+"\n")
            cid = self.create_cell_frame_class()
            cid.read_raw(frame_name = frame,
                         cell_seg_data_file=data,
                         score_data_file=score,
                         binary_seg_image_file=binary_seg_maps,
                         component_image_file=component_image,
                         inform_analysis_dict=inform_analysis_dict,
                         verbose=verbose,
                         require_component=require_component,
                         require_score=require_score,
                         skip_segmentation_processing=skip_segmentation_processing,
                         dry_run=dry_run
                         )
            if verbose: sys.stderr.write("setting tumor and stroma and margin\n")
            cid.set_line_area(margin,tumor,steps=steps,verbose=verbose)
            frame_id = cid.id
            self._frames[frame_id]=cid
            frames.append({'frame_id':frame_id,'frame_name':frame,'frame_path':absdir})
            if verbose: sys.stderr.write("finished tumor and stroma and margin\n")
        self._key = pd.DataFrame(frames)
        self._key.index.name = 'db_id'
        self.sample_name = sample_name         

class CellFrameInFormLineArea(CellFrameInForm):
    def __init__(self):
        super().__init__()
        ### Define extra InForm-specific data tables
        #self.data_tables['custom_images'] = {'index':'db_id',
        #         'columns':['custom_label','image_id']}
        for x in self.data_tables.keys():
            if x in self._data: continue
            self._data[x] = pd.DataFrame(columns=self.data_tables[x]['columns'])
            self._data[x].index.name = self.data_tables[x]['index']
    def set_line_area(self,line_image,area_image,steps=-1,verbose=False):
        if steps <0:
            raise ValueError("Steps must be set and must be greater than or equal to zero")
        if verbose:
            sys.stderr.write("Initialize numpy arrays\n")
        processed_image = self.get_image(self.processed_image_id).astype(np.uint8)
        zero_binary = processed_image&(~processed_image)
        drawn_binary = np.zeros(self.shape)
        if verbose:
            sys.stderr.write("Done initializing numpy arrays\n")
        image_id2= uuid4().hex
        if verbose:
            sys.stderr.write("Reading tumor image\n")
        area_binary = read_tiff_stack(area_image)[0]['raw_image']
        if verbose:
            sys.stderr.write("Binarizing tumor image\n")
        area_binary = make_binary_image_array(area_binary)
        full_tumor = area_binary&processed_image
        if verbose:
            sys.stderr.write("Done reading tumor image\n")
        self._images[image_id2] = area_binary
        if line_image is not None and os.path.exists(line_image):
            if verbose:
                sys.stderr.write("Reading line image\n")
            drawn_binary = read_tiff_stack(line_image)[0]['raw_image']
            if verbose:
                sys.stderr.write("Binarizing tumor image\n")
            drawn_binary = make_binary_image_array(drawn_binary)
            if verbose:
                sys.stderr.write("Done reading line image\n")
        else:
            # specialcase of no line
            df = pd.DataFrame({'mask_label':['Area'],'image_id':[image_id2]})
            df.index.name = 'db_id'
            self.set_data('mask_images',df)
            if verbose:
                sys.stderr.write("Determine region masks for tumor only\n")
            inner_tumor_binary = full_tumor&processed_image
            inner_margin_binary = zero_binary
            outer_margin_binary = zero_binary
            outer_stroma_binary = zero_binary
            undefined_binary = (~full_tumor)&processed_image
            if verbose:
                sys.stderr.write("Skipping the invasive margin since there is none present.\n")
            d = {'InnerTumor':inner_tumor_binary,
             'InnerMargin':inner_margin_binary,
             'OuterMargin':outer_margin_binary,
             'OuterStroma':outer_stroma_binary,
             'Undefined':undefined_binary
            }
            if verbose:
                sys.stderr.write("Setting regions\n")
            self.set_regions('InFormLineArea',d,description="Tumor stroma interface set from a mask and optional drawn line.")
            if verbose:
                sys.stderr.write("Regions are set\n")
            return d
        image_id1 = uuid4().hex
        self._images[image_id1] = drawn_binary

        df = pd.DataFrame({'mask_label':['Drawn','Area'],'image_id':[image_id1,image_id2]})
        df.index.name = 'db_id'
        self.set_data('mask_images',df)

        if verbose:
            sys.stderr.write("Execute binary dialation for steps="+str(steps)+" pixels\n")
        grown = binary_image_dilation(drawn_binary,steps=steps)
        if verbose:
            sys.stderr.write("Determine region masks for margin and tumor\n")
        margin_binary = grown&processed_image
        inner_tumor_binary = (full_tumor&(~margin_binary))&processed_image
        inner_margin_binary = (full_tumor&(margin_binary))&processed_image
        outer_margin_binary = (margin_binary&(~inner_margin_binary))&processed_image
        outer_stroma_binary = (~(inner_tumor_binary|margin_binary))&processed_image
        undefined_binary = zero_binary

        d = {'InnerTumor':inner_tumor_binary,
             'InnerMargin':inner_margin_binary,
             'OuterMargin':outer_margin_binary,
             'OuterStroma':outer_stroma_binary,
             'Undefined':undefined_binary
            }
        if verbose:
            sys.stderr.write("Setting regions\n")
        self.set_regions('InFormLineArea',d,description="Tumor stroma interface set from a mask and optional drawn line.")
        if verbose:
            sys.stderr.write("Regions are set\n")
        return d

