import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import trange
import csv
import argparse

final_err = 0
##############
exp_name = 'Final_CLN_Cond_Conv'
##############
#log_dir = "/home/ziqi/Desktop/iros_final_results_analysis/Final_Results/" + exp_name + "/logs/1/" + "best_val" + "/" 
log_dir = '/home/jason/Desktop/Dropout_Evals/Final_CondConv/logs/LR05/'
for fname in ["predictions29.txt", "predictions70.txt","predictions73.txt","predictions74.txt","predictions77.txt"]:
    results = [] 
    # errors = []
    det_x = []
    det_y = []
    gt_x = []
    gt_y = []

    with open(log_dir + fname, "r") as fp:
        entries = fp.readlines()
        for line in entries:
            if "Dets" in line:
                continue
            new_line = ""
            for letter in line:
                if letter not in "[]":
                    new_line += letter
            assert len(new_line.strip().split()) == 4
            det_x.append( eval(new_line.strip().split()[0]))
            det_y.append( eval(new_line.strip().split()[1]))
            gt_x.append( eval(new_line.strip().split()[2]))
            gt_y.append( eval(new_line.strip().split()[3]))
            # error = ((det_x - gt_x)**2 + (det_y - gt_y)**2)**0.5
            # results.append([(det_x, det_y), (gt_x, gt_y)])
            # errors.append(error)
    det_x = np.array(det_x)
    det_y = np.array(det_y)

    offset = 0 if ("29" in fname or "70" in fname) else 9
    gt_x_off = np.array(gt_x[offset:])
    gt_y_off = np.array(gt_y[offset:])
    error = 0
    for i in range(len(gt_x)-offset):
        error += ((det_x[i] - gt_x_off[i])**2 + (det_y[i] - gt_y_off[i])**2)**0.5
    print("Filename {}, error {}".format(fname, error/i))
    final_err += error/i
print(final_err/5.0)


# for offset in range(0,21):
#     gt_x_off = np.array(gt_x[offset:])
#     gt_y_off = np.array(gt_y[offset:])
#     error = 0
#     for i in range(len(gt_x)-offset):
#         error += ((det_x[i] - gt_x_off[i])**2 + (det_y[i] - gt_y_off[i])**2)**0.5
#     print("Offset {}, error {}".format(offset, error/i))
# import ipdb; ipdb.set_trace()
# A = np.zeros((5,1))
# agg_error = []
# for i in range(5):
#     A[i] = np.mean(errors[i])
#     for each in errors[i]:
#         agg_error.append(each)
# with open("temp.csv", "w") as fp:
#     for each in A:
#         print(each[0],end=",",file=fp)

# print(np.mean(A))

# with open(test_name, 'w', newline='') as csvfile:
#     testwriter = csv.writer(csvfile, delimiter=',',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     testwriter.writerow(agg_error)

