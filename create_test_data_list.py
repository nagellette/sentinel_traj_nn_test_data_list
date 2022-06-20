from utils.train_test_splitter import TrainTestValidateSplitter
import pandas as pd

sample_path_mont = "/home/ngengec/Desktop/final_data/mtl_final_small/montreal_20160720_B02_small.tif"
sample_path_ist = "/home/ngengec/Desktop/final_data/ist_data/T35TPF_20191109T090201_B02_10m_clipped.tif"
sample_mask_path_ist = "/home/ngengec/Desktop/final_data/ist_data/istanbul-internal_2019_landmass_mask_prj.tif"
train_size = 0.6
test_size = 0.2
validate_size = 0.2
image_dims = (512, 512)
augment = 45
overlap = 0.2
seed = 10
coverage_check_mont = False
coverage_check_ist = True
label_threshold = 0.01
check_label = True

data_splitter_mont = TrainTestValidateSplitter(sample_file=sample_path_mont,
                                               mask_file=False,
                                               train=train_size,
                                               test=test_size,
                                               validation=validate_size,
                                               dim=image_dims,
                                               augment=augment,
                                               overlap=overlap,
                                               seed=seed,
                                               check_coverage=coverage_check_mont,
                                               label_threshold=label_threshold,
                                               check_label=check_label
                                               )

train_mont, test_mont, val_mont = data_splitter_mont.get_train_test_validation()
df_test_mont = pd.DataFrame(test_mont)
df_test_mont.to_csv("./test_list_mont.csv", index=False)

data_splitter_ist = TrainTestValidateSplitter(sample_file=sample_path_ist,
                                              mask_file=sample_mask_path_ist,
                                              train=train_size,
                                              test=test_size,
                                              validation=validate_size,
                                              dim=image_dims,
                                              augment=augment,
                                              overlap=overlap,
                                              seed=seed,
                                              check_coverage=coverage_check_ist,
                                              label_threshold=label_threshold,
                                              check_label=check_label
                                              )

train_ist, test_ist, val_ist = data_splitter_ist.get_train_test_validation()
df_test_ist = pd.DataFrame(test_ist)
df_test_ist.to_csv("./test_list_ist.csv", index=False)
