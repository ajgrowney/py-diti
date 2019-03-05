import csv

def writeResults(cancer_res, nocancer_res, filename):
    f = open(filename,'w')
    with f:
        fieldnames_f = ['f_r_skew', 'f_g_skew', 'f_b_skew', 'f_r_kurt', 'f_g_kurt', 'f_b_kurt', 'f_r_image_mean', 'f_g_image_mean','f_b_image_mean', 'f_r_left_mean', 'f_g_left_mean', 'f_b_left_mean', 'f_r_right_mean', 'f_g_right_mean', 'f_b_right_mean']
        fieldnames_fc = ['fc_r_skew', 'fc_g_skew', 'fc_b_skew', 'fc_r_kurt', 'fc_g_kurt', 'fc_b_kurt', 'fc_r_image_mean', 'fc_g_image_mean','fc_b_image_mean', 'fc_r_left_mean', 'fc_g_left_mean', 'fc_b_left_mean', 'fc_r_right_mean', 'fc_g_right_mean', 'fc_b_right_mean']
        fieldnames = fieldnames_f + fieldnames_fc + ['cancer']
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(len(cancer_res)):
            writer.writerow(cancer_res[i]+['Y'])
        for i in range(len(nocancer_res)):
            writer.writerow(nocancer_res[i]+['N'])
