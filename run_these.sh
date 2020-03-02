python main_sdtw.py --loss rae --save ../LSTNetLogs/elec-l1-f-24 --data data/electricity.txt
python main_sdtw.py --loss rse --save ../LSTNetLogs/elec-l2-f-24 --data data/electricity.txt
python main_sdtw.py --loss sdtw --save ../LSTNetLogs/elec-sdtw-f-24 --data data/electricity.txt
python main_sdtw.py --horizon 12 --loss sdtw --save ../LSTNetLogs/elec-sdtw-f-12 --data data/electricity.txt
python main_sdtw.py --horizon 6 --loss sdtw --save ../LSTNetLogs/elec-sdtw-f-6 --data data/electricity.txt
python main_sdtw.py --horizon 3 --loss sdtw --save ../LSTNetLogs/elec-sdtw-f-3 --data data/electricity.txt
