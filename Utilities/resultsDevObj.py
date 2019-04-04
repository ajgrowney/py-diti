import csv

class resultsObj:
    def __init__(self):
        self["skews"]= {
            "r_f": [], "r_fc": [], "g_f": [], "g_fc": [], "b_f": [], "b_fc": []
        },
        self["kurtosis"]= {
            "r_f": [], "r_fc": [], "g_f": [], "g_fc": [], "b_f": [], "b_fc": []
        },
        self["means"]= {
            "total_f": {
                "r": [],
                "g": [],
                "b": []
            },
            "total_fc": {
                "r": [],
                "g": [],
                "b": []
            },
            "left_f": {
                "r": [],
                "g": [],
                "b": []
            },
            "left_fc": {
                "r": [],
                "g": [],
                "b": []
            },
            "right_f": {
                "r": [],
                "g": [],
                "b": []
            },
            "right_fc": {
                "r": [],
                "g": [],
                "b": []
            }
         }


def writeResultsToCsv(cancer_obj,nocancer_obj, filename):
    f = open(filename,'w')

    with f:
        fieldnames = ['skews_r_f','skews_r_fc','skews_g_f','skews_g_fc', 'skews_b_f','skews_b_fc','kurtosis_r_f','kurtosis_r_fc','kurtosis_g_f','kurtosis_g_fc', 'kurtosis_b_f','kurtosis_b_fc', 'mean_total_f_r', 'mean_total_f_g', 'mean_total_f_b', 'mean_total_fc_r', 'mean_total_fc_g', 'mean_total_fc_b', 'cancer']
        writer = csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(cancer_obj["means"]["total_f"]["r"])):
            writer.writerow(
                {
                    'skews_r_f': cancer_obj["skews"]["r_f"][i],
                    'skews_r_fc': cancer_obj["skews"]["r_fc"][i],
                    'skews_g_f': cancer_obj["skews"]["g_f"][i],
                    'skews_g_fc': cancer_obj["skews"]["g_fc"][i],
                    'skews_b_f': cancer_obj["skews"]["b_f"][i],
                    'skews_b_fc': cancer_obj["skews"]["b_fc"][i],
                    'kurtosis_r_f': cancer_obj["kurtosis"]["r_f"][i],
                    'kurtosis_r_fc': cancer_obj["kurtosis"]["r_fc"][i],
                    'kurtosis_g_f': cancer_obj["kurtosis"]["g_f"][i],
                    'kurtosis_g_fc': cancer_obj["kurtosis"]["g_fc"][i],
                    'kurtosis_b_f': cancer_obj["kurtosis"]["b_f"][i],
                    'kurtosis_b_fc': cancer_obj["kurtosis"]["b_fc"][i],
                    'mean_total_f_r': cancer_obj["means"]["total_f"]["r"][i],
                    'mean_total_fc_r': cancer_obj["means"]["total_fc"]["r"][i],
                    'mean_total_f_g': cancer_obj["means"]["total_f"]["g"][i],
                    'mean_total_fc_g': cancer_obj["means"]["total_fc"]["g"][i],
                    'mean_total_f_b': cancer_obj["means"]["total_f"]["b"][i],
                    'mean_total_fc_b': cancer_obj["means"]["total_fc"]["b"][i],
                    'cancer': 'Y'
                }
            )

        for j in range(len(nocancer_obj["means"]["total_f"]["r"])):
            writer.writerow(
                {
                    'skews_r_f': nocancer_obj["skews"]["r_f"][j],
                    'skews_r_fc': nocancer_obj["skews"]["r_fc"][j],
                    'skews_g_f': nocancer_obj["skews"]["g_f"][j],
                    'skews_g_fc': nocancer_obj["skews"]["g_fc"][j],
                    'skews_b_f': nocancer_obj["skews"]["b_f"][j],
                    'skews_b_fc': nocancer_obj["skews"]["b_fc"][j],
                    'kurtosis_r_f': nocancer_obj["kurtosis"]["r_f"][j],
                    'kurtosis_r_fc': nocancer_obj["kurtosis"]["r_fc"][j],
                    'kurtosis_g_f': nocancer_obj["kurtosis"]["g_f"][j],
                    'kurtosis_g_fc': nocancer_obj["kurtosis"]["g_fc"][j],
                    'kurtosis_b_f': nocancer_obj["kurtosis"]["b_f"][j],
                    'kurtosis_b_fc': nocancer_obj["kurtosis"]["b_fc"][j],
                    'mean_total_f_r': nocancer_obj["means"]["total_f"]["r"][j],
                    'mean_total_fc_r': nocancer_obj["means"]["total_fc"]["r"][j],
                    'mean_total_f_g': nocancer_obj["means"]["total_f"]["g"][j],
                    'mean_total_fc_g': nocancer_obj["means"]["total_fc"]["g"][i],
                    'mean_total_f_b': nocancer_obj["means"]["total_f"]["b"][i],
                    'mean_total_fc_b': nocancer_obj["means"]["total_fc"]["b"][i],
                    'cancer': 'N'
                }
            )
