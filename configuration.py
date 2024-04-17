import sys,os
import getpass
from pathlib import Path

if getpass.getuser() in ('samuel.garcia', 'valentin.ghibaudo','baptiste.balanca', 'gwendan.percevault') and  sys.platform.startswith('linux'):
    # base_folder = '/home/valentin/smb4k/CRNLDATA/crnldata/tiger/baptiste.balanca/Neuro_rea_monitorage/'
    base_folder = '/crnldata/tiger/baptiste.balanca/Neuro_rea_monitorage/'

elif sys.platform.startswith('win') and getpass.getuser() in ('baptiste.balanca'):
    base_folder = 'N:/baptiste.balanca/Neuro_rea_monitorage/'
    
elif sys.platform.startswith('win') and getpass.getuser() in ('gwenp'):
    base_folder = 'n:/tiger/baptiste.balanca/Neuro_rea_monitorage/'
    
elif sys.platform.startswith('darwin') and getpass.getuser() in ('gwendanpercevault'): 
    base_folder = 'smb://10.69.168.1/tiger/baptiste.balanca/Neuro_rea_monitorage/'

base_folder = Path(base_folder)
data_path = base_folder / 'raw_data'
