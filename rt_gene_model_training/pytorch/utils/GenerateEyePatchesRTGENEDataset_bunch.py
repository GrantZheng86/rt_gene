import os
import sys
import shutil


if __name__ == "__main__":
    script_desc = open('GenerateEyePatchesRTGENEDataset.py')
    script_main = script_desc.read()

    dataset_location = '/media/grant/Dataset/Gaze_dataset'
    sample_directory_list = os.listdir(dataset_location)
    extracted_patch_dir = 'inpainted/Processed_eyes'
    counter = 0

    for each_dir in sample_directory_list:
        print('process sample {} out of {}'.format(counter, len(sample_directory_list)))
        inpainted_face_dir = os.path.join(dataset_location, each_dir, 'inpainted/face_after_inpainting')
        save_dir = os.path.join(dataset_location, each_dir, extracted_patch_dir)

        isExist = os.path.exists(save_dir)
        if isExist:
            shutil.rmtree(save_dir)
            print("Old directory removed for sample {}".format(counter))
        isExist = os.path.exists(save_dir)
        print(isExist)
        if not isExist:
            os.makedirs(save_dir)

        sys.argv = ["GenerateEyePatchesRTGENEDataset.py", inpainted_face_dir, '--output_path', save_dir]
        exec(script_main)
        counter += 1

    script_desc.close()
