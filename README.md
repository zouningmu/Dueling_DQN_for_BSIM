# BSIM parameter extraction method based on RL method
This project primarily utilizes the Dueling DQN method to automate the extraction of BSIM (Berkeley Short-channel IGFET Model) parameters.

## Installation Instructions
### Environmental Dependency
```
gymnasium==0.28.1
matplotlib==3.7.2
numpy==1.24.3
pandas==2.0.3
setuptools==75.1.0
torch==2.4.1
tqdm==4.66.5
```

### Installation Process

a. （optional）Create and activate a conda virtual environment.
```bash
conda create -n bsim python=3.8
conda activate bsim
```

b. Clone the BSIM code repository.
```bash
git clone 
```

c. Install the BSIM environment dependencies.
```bash
pip install -r requirements.txt
```

d. Test your net table and model card
You can test your netlist and model card for errors in your local HSPICE environment. All BSIM parameters are stored in the model card.

After testing, place the model card and netlist in the test_data directory, and place the reference data (from TACD simulation or wafer fab silicon data) in the test_data directory.
```
BSIM/test_data/
├── your_netlisy.sp
├── your_model.l
└── your_ref_data.dat
```

e. HSPICE installation directory
Based on your HSPICE installation directory, modify the directory that calls the simulator. The statement that calls HSPICE is 
```bash
result = subprocess.run(command, shell=True, cwd=r'{your HSPICE installation directory}', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

f. Configure your parameter extraction task:
In the param_list.xls file, set the specific parameters to be extracted, as well as their upper limits, lower limits, and initial values.

In the task_list.xls file, set the netlist, reference data file path model card, simulation result save path, curve types involved (e.g., IV, CV), and task batch. 
Through the task batch and curve types, you can set different parameter extraction tasks and parameter group extraction.

## Training Process

a.  Use CPU for training
```bash
python project/gym_bsim/main.py
```
If you use a GPU for training, you need to modify DQN_training.py. 
However, the performance improvement from GPU training will not be significant.


b. Training details
1. The ultimate goal of this project is to extract a set of BSIM parameters that minimize the difference between IV and CV simulation results and real data. Complete reinforcement learning training is not necessary, so the training cycle can be greatly reduced. 
2. Each time the environment is reset, it is not necessary to start from the same set of parameters, but rather from the parameter combination with the smallest error from the previous cycle.
3. Document parsing currently only supports two types of HSPICE simulation result files (.lis or .csv files). More types can be customized in BSIM\project\gym_bsim\gym_bsim\envs\bsim.py.
4. There are no requirements for setting the parameter range. Generally speaking, the smaller the parameter range, the faster it is to find a suitable solution. Repeatedly narrowing the parameter range is very helpful for the results.


## Contact Persons
nzou@nju.edu.cn
wenjunchen@smail.nju.edu.cn