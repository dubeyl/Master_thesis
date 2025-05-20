#### How to use the automation
Hardware prep:
- plug in laser light (and turn it on), connection to main camera, connection to basler camera, arduino
Software prep:
- navigate to correct folder: cd /flymuscle-control/automation_LD
- activate environment: conda activate automation_LD_env
- update target folders in LD_config.yaml (muscle folders, base folder for all recordings and output folder for transfer to server)
- launch the automation: python master_script.py

#### to solve the basler camera admin problem
- sudo nano /etc/udev/rules.d/99-basler-usb.rules
- SUBSYSTEM=="usb", ATTR{idVendor}=="2676", ATTR{idProduct}=="ba02", MODE="0666" 
##### then press ctrl+o, enter, ctrl+x
- sudo udevadm control --reload-rules
(-) sudo udevadm trigger
- ls -l /dev/bus/usb/002/009
