import dicom
import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir=''
patients = os.listdir(data_dir)
labels_df = pd.read_csv("stage1_labels.csv", index_col=0)


labels_df.head()

for patient in patients[:1]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    print(len(slices), label)
    print(slices[0].pixel_array_shape, len(slices))

    plt.imshow(slices[0].pixel_array)
    plt.show()


