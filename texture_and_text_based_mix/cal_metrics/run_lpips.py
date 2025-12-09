from lpips_offical import cal_lpips
from tqdm import tqdm

# ids=[(1,151),(151,301),(301,451),(451,601),(601,751),(751,901),(901,933)]
ids=[[0,400]]

for i in tqdm(range(len(ids)),position=0):
    my_cur_lpips=cal_lpips(ids[i][0],ids[i][1])
    if i==0:
        my_avg_lpips=my_cur_lpips
        # compare_avg_lpips=compare_cur_lpips
    else:
        my_avg_lpips=(my_cur_lpips+my_avg_lpips)/2
        # compare_avg_lpips=(compare_cur_lpips+compare_avg_lpips)/2

print("total avgs: ",my_avg_lpips)
