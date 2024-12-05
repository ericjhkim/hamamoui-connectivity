import tools as tools

N_list = []
P_THRESH_list = []
pr_list = []

paths = ["data_20241204_150855_5_10.h5","data_20241204_123442_6_10.h5","data_20241204_115953_7_10.h5","data_20241204_183635_8_10.h5"]

for path in paths:
    N, P_THRESH, phis, matching_frequencies, pr = tools.load_data(f"data/{path}")
    N_list.append(N)
    P_THRESH_list.append(P_THRESH)
    pr_list.append(pr)

tools.plot_multi_data(N_list, P_THRESH_list, phis, pr_list)
