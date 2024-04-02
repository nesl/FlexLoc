cd ~/Desktop/Final_Final_Evals/Final_CLN_Resnet
data_root=/mnt/ssd_8t/jason/IROS/Cached_Datasets/Final_Final_Dataset
for test_view in 29 70 73 74 77
do

    mv ${data_root}/test${test_view} ${data_root}/test
    log_folder=AllMods
    python3 batch_test.py --folder ${log_folder} --checkpoint best_val.pt
    mv ./logs/${log_folder}/predictions.txt ./logs/${log_folder}/predictions${test_view}.txt
    mv ./logs/${log_folder}/test_loss.txt ./logs/${log_folder}/test_loss${test_view}.txt
    
    mv ${data_root}/test ${data_root}/test${test_view}
done
