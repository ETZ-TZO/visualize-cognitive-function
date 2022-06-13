import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import SimpleITK as sitk
import re
import os
from scipy import ndimage
from tqdm import tqdm

# Dir with niftis (segmentations)
niftidir = "/home/sander/Documents/nnUNet_data/nnUNet_raw/nnUNet_raw_data/Task201_ETZDataPaul/raw_sander/labels_ones/"
# Where we want to save our visualisations
outdir = "/home/sander/Documents/nnUNet_data/nnUNet_raw/nnUNet_raw_data/Task201_ETZDataPaul/raw_sander/paper_paul/"
# spss file storing our z-scores and impairment scores
scoresfile = "/home/sander/Documents/sander_nps_basefiles2/tumor_features/20220303 Total WeCare.sav"


def calculate_visualisation(df, for_vars, run_name) -> None:
    '''
    Loops over all nifti files (segmentations) in our niftidir.
    niftidir is expected to be a flat dir with an integer in the file name representing the patient id. (e.g 'patient_012.nii.gz')
    The tumor is expected to be represented as 1s with all other pixels being 0.
    All nifties need to be registered to a common template
    Visualizations for all variables are calculated simultaneously.
    '''
    files = [x for x in os.walk(niftidir)][0][2]

    # Empty results which we will add the results to
    template_image = sitk.ReadImage(niftidir + files[0])
    shape = sitk.GetArrayFromImage(template_image).shape
    all_results = np.zeros(shape)
    all_results = np.tile(all_results, (len(for_vars), 1, 1, 1))  # Z-score / impairment visualisations
    tumors_per_pixel = all_results.copy()  # Number of tumors per pixel
    tumors_per_pixel_for_vis_frontal = np.zeros(shape)  # Visualisation number of frontal tumors
    tumors_per_pixel_for_vis_nonfrontal = np.zeros(shape)  # Visualisation number of non-frontal tumors

    # Paths and names for saving
    out_names = [item + '_' + run_name for item in for_vars]
    output_files = [outdir + out_name for out_name in out_names]

    # Iteratively load files and add the scores to the result
    for file in tqdm(files):
        if file.endswith('.nii.gz'):
            print(f'Calculating for: {file}')
            id = int(re.sub("[^0-9]", "", file))

            if not int(id) in df['id_code'].values:
                print('no data for: ', id)
                continue

            # The the scores for this patient
            vals = df[df['id_code'] == int(id)][for_vars].values[0].astype(float)
            print(vals)

            img_in = sitk.ReadImage(niftidir + file)
            img_npy = sitk.GetArrayFromImage(img_in)
            tumor_size = np.sum(img_npy)

            if tumor_size == 0:
                print("No tumor in segmentation")
                continue

            to_add = np.multiply.outer(vals, img_npy)
            to_add = np.nan_to_num(to_add)

            # Only count segmentation for scores that are not nan
            only_when_notnan = np.multiply.outer(~np.isnan(vals), img_npy != 0)
            only_when_notnan = np.nan_to_num(only_when_notnan)
            tumors_per_pixel = tumors_per_pixel + only_when_notnan

            all_results = all_results + to_add

            # Count tumors for the seperate visualisation on tumor location
            frontal = df[df['id_code'] == int(id)]['frontal'].values[0] == 'yes'
            nonfrontal = df[df['id_code'] == int(id)]['frontal'].values[0] == 'no'
            tumors_per_pixel_for_vis_nonfrontal = tumors_per_pixel_for_vis_nonfrontal + img_npy * nonfrontal
            tumors_per_pixel_for_vis_frontal = tumors_per_pixel_for_vis_frontal + img_npy * frontal

    # Devide results by tumors per pixel
    nan_array = np.zeros_like(all_results)
    nan_array[:] = np.nan
    all_results_corrected = np.divide(all_results, tumors_per_pixel, out=nan_array,
                                      where=(tumors_per_pixel) != 0)

    # Create an output per variable
    for result, output_file in zip(all_results_corrected, output_files):
        img_out_itk = sitk.GetImageFromArray(result)
        img_out_itk.SetOrigin(template_image.GetOrigin())
        img_out_itk.SetDirection(template_image.GetDirection())
        img_out_itk.SetSpacing(template_image.GetSpacing())

        sitk.WriteImage(img_out_itk, output_file + '_divided_by_n_tumors.nii.gz')

    for result, output_file in zip(all_results, output_files):
        img_out_itk = sitk.GetImageFromArray(result)
        img_out_itk.SetOrigin(template_image.GetOrigin())
        img_out_itk.SetDirection(template_image.GetDirection())
        img_out_itk.SetSpacing(template_image.GetSpacing())

        sitk.WriteImage(img_out_itk, output_file + '_raw.nii.gz')

    # Create visualisation for tumor location (independend of the z-scores/impairment)
    print('Creating tumors per pixel frontal vs non-frontal')
    img_out_itk = sitk.GetImageFromArray(tumors_per_pixel_for_vis_nonfrontal)
    img_out_itk.SetOrigin(template_image.GetOrigin())
    img_out_itk.SetDirection(template_image.GetDirection())
    img_out_itk.SetSpacing(template_image.GetSpacing())
    sitk.WriteImage(img_out_itk, outdir + 'tumors_per_pixel_for_vis_nonfrontal.nii.gz')

    img_out_itk = sitk.GetImageFromArray(tumors_per_pixel_for_vis_frontal)
    img_out_itk.SetOrigin(template_image.GetOrigin())
    img_out_itk.SetDirection(template_image.GetDirection())
    img_out_itk.SetSpacing(template_image.GetSpacing())
    sitk.WriteImage(img_out_itk, outdir + 'tumors_per_pixel_for_vis_frontal.nii.gz')


def main() -> None:
    # Load file. Expects one row per patient. Patients are identified by the 'id_code' (int) column.
    # All other columns can contain z-scores or 'Non-impaired or impaired'
    df = pd.read_spss(scoresfile)
    df = df.rename({'ID_code': 'id_code'}, axis=1)
    df['id_code'] = df['id_code'].astype(int)

    # Our variable names
    for_vars_zsc = ['Zsc_verbalmem_T0',
                    'Zsc_vismem_T0',
                    'Zsc_motorspeed_T0',
                    'Zsc_FTT_T0',
                    'Zsc_CPT_reactietijd_T0',
                    'Zsc_SDC_T0',
                    'Zsc_SAT_T0',
                    'Zsc_EF_T0',
                    'Zsc_Stroop_simple_T0',
                    'Zsc_Stroop_IIIrt_T0',
                    'Zsc_Stroop_interferentie_T0',
                    'Zsc_fluency_T0',
                    'Zsc_DSBW_T0',
                    'Zsc_DSFW_T0']

    for_vars_prob = ['Zsc_verbalmem_cutoff',
                     'Zsc_vismem_cutoff',
                     'Zsc_FTT_cutoff',
                     'Zsc_SDC_cutoff',
                     'Zsc_StroopI_cutoff',
                     'Zsc_CPT_cutoff',
                     'Zsc_DSFW_cutoff',
                     'Zsc_NonExecutive_reduced',
                     'Zsc_StroopIII_cutoff',
                     'Zsc_Stroop_Interfer_cutoff',
                     'Zsc_SAT_cutoff',
                     'Zsc_VerbalFluency_cutoff',
                     'Zsc_DSBW_cutoff',
                     'Zsc_Executive_reduced',
                     'Zsc_impaired_v2']

    # Convert impaired, non-impaired to int
    for var in for_vars_prob:
        df[var] = df[var].map({'Normal': 0,
                               'Impaired': 1,
                               'Non-impaired': 0})

    print('Working on visualisations')
    calculate_visualisation(df, for_vars_zsc + for_vars_prob, 'paper')


if __name__ == '__main__':
    print('Calculating visualisations')
    main()
    print('Done')
