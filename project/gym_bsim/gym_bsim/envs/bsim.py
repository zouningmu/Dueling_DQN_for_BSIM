import copy
import logging
import os
import re
import subprocess
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from matplotlib import pyplot as plt
import shutil

plt.rc('font', family='Times New Roman')
logger = logging.getLogger(__name__)


def get_IV(file_path, curve_type, condition_value):
    fit_bool = False
    end_bool = False
    fit_data = {}
    fit_list = []
    p_values = []
    with open(file_path, 'r') as file:
        for line in file:
            if line is None or len(line.replace('\n', '')) == 0:
                continue
            line = line.replace('\n', '')
            # print line
            if 'group' in line and 'condition' in line:
                if fit_bool and end_bool:
                    break
                line_data = re.split(r",(?![^(]*\))", re.sub(r'[{}]', '', line))
                line_map = {k: v for k, v in (re.split(r"=(?![^(]*\))", x1) for x1 in line_data)}
                _group = line_map.get('group')
                _condition_data = str(line_map.get('condition')).replace("(", "").replace(")", '')
                condition_map = {k: v for k, v in (x1.split("=") for x1 in _condition_data.split(','))}

                del condition_map['ref_vs']
                _condition_name = ''
                _condition_value = ''
                for key in condition_map:
                    _condition_name = key
                    _condition_value = condition_map[key]

                if curve_type == _group and float(condition_value) == float(_condition_value):
                    fit_bool = True
                    end_bool = True
                    _p = str(line_map.get('p'))
                    _psplit = _p.split("(")
                    _pname = _psplit[0]
                    p_values = _psplit[1].replace(')', '').split(',')

                    _x_name = line_map.get('x')
                    fit_data = {
                        'group': _group,
                        'p_name': _pname,
                        # 'p_value': _p_value,
                        'x_name': _x_name,
                        'y_name': re.sub(r'[()]', '', line_map.get('y')),
                        'device': line_map.get('device'),
                        'condition': line_map.get('condition'),
                        'condition_name': _condition_name,
                        'condition_value': _condition_value
                    }

            elif fit_bool and end_bool:
                data = re.split(r'\s+', line.strip())
                for i in range(1, len(data)):
                    _fit = fit_data.copy()
                    _fit['x_value'] = data[0]
                    _fit['p_value'] = p_values[i - 1]
                    _fit['y_value'] = data[i]
                    fit_list.append(_fit)
    return fit_list

def get_CV(file_path, curve_type):
    fit_bool = False
    end_bool = False
    fit_data = {}
    fit_list = []
    with open(file_path, 'r') as file:
        for line in file:
            if line is None:
                continue
            line = line.replace('\n', '')
            if len(line) == 0 or 'condition' in line:
                continue
            if 'Page' in line:
                if fit_bool and end_bool:
                    break
                _page = line.replace('Page', '').split("{")
                line_data = re.split(r",", re.sub(r'[()]', '', _page[0]))
                #line_map = {k: v for k, v in (re.split(r"=(?![^(]*\))", x1) for x1 in line_data)}
                line_map = {k: v for k, v in (re.split(r"=", x1) for x1 in line_data)}
                _group = line_map.get('name')
                if curve_type == _group:
                    fit_bool = True
                    end_bool = True
                    fit_data = {
                        'group': _group,
                        'x_name': line_map.get('x'),
                        'y_name': line_map.get('y'),
                        'device': '{' + _page[1]
                    }
            elif fit_bool and end_bool:
                if 'curve' in line:
                    fit_data['curve'] = re.sub(r'[{}]', '', line.replace('curve', ''))
                else:
                    _line = line.split(',')
                    _data = fit_data.copy()
                    _data['x_value'] = _line[0]
                    _data['y_value'] = _line[1]
                    fit_list.append(_data)

    return fit_list

def extract_target(file_path,curve_type):
    if curve_type in('cgc_vgs_vbs','cgg_vgs'):
        fit_data = get_CV(file_path, curve_type)
    elif curve_type in('Id_Vg_A','Id_Vg_B','Id_Vd_A','Id_Vd_B'):
        fit_data = get_IV(file_path,curve_type, -0)

    df = pd.DataFrame(fit_data)
    df_fit=df[['x_value','y_value']].rename(columns={'x_value':'x','y_value':'y'})
    df_fit['x'] = pd.to_numeric(df_fit['x'], errors='coerce')
    df_fit['y'] = pd.to_numeric(df_fit['y'], errors='coerce').abs()
    # if df_fit['y'][0] > 1e-6:
    #     df_fit['y'] = df_fit['y'] * 1e-12
    print(df_fit.head())
    return df_fit

def extract_reference(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('v1'):
            data_start = i + 1
            break

    data = []
    for line in lines[data_start:]:
        line = line.strip()
        if line:
            parts = line.split(',')
            if len(parts) >= 2:
                data.append(parts[:2])

    df = pd.DataFrame(data, columns=['x', 'y'])

    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce').abs()
    df.dropna(subset=['x', 'y'], inplace=True)

    return df


def reset_modelcard(state, model_path, parameter):
    try:
        with open(model_path, 'r') as file:
            content = file.readlines()

        updated_content = []
        param_found = {key: False for key in parameter.keys()}

        for line in content:
            _data = line.replace('\n', '').split(' ')
            index = 0
            fitting_bool = False
            _param = ''
            while index < len(_data):
                if _data[index] in parameter.keys():
                    fitting_bool = True
                    _param = _data[index]
                    indic = parameter[_data[index]]
                    indic = indic['index']
                    param_found[_data[index]] = True
                elif _data[index].strip() != '=' and fitting_bool:
                    fitting_bool = False
                    _data[index] = f'{state[indic]}'
                index += 1
            updated_content.append(' '.join(_data) + '\n')

        with open(model_path, 'w') as file:
            file.writelines(updated_content)

        missing_params = [param for param, found in param_found.items() if not found]
        if missing_params:
            print(f"Error: The following parameters were not found in the model file: {', '.join(missing_params)}")

    except FileNotFoundError:
        print(f"Error: Model file {self.model_path[0]} not find.")
    except Exception as e:
        print(f"Error: An exception occurred while updating the model file. - {e}")


def calculate_rms(df1, df2):
    if len(df1) != len(df2):
        raise ValueError("The lengths of the input DataFrames must be the same")
    squared_errors = ((df1['y'].values - df2['y'].values) / df1['y'].values) ** 2
    rms = np.sqrt(np.mean(squared_errors))

    return rms


class BSIMEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50,
        'render_fps': 50,
    }

    def __init__(self, first_state, high, low, task_batch_path, target, fitting, model, sp, parameter, t_limit=100, step_size=10.0, train_model=True,
                 curve_type=None):
        self.curve_type = curve_type
        self.high = high
        self.low = low

        self.first_state = first_state[:]
        self.action_space = spaces.Discrete((max(param_data['index'] for param_data in parameter.values())+1)*2)
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float64)

        self.task_batch_path = task_batch_path
        self.fitting_path = fitting
        self.model_path = model
        self.parameter = parameter

        self.seed(0)
        self.viewer = None
        self.state = None
        self.step_size = step_size
        self.sp = sp

        self.command = []
        for sp, sp_info in self.sp.items():
            self.command.append(rf'.\hspice {sp_info} -o {self.task_batch_path[sp]}\results\hspice_res')

        self.curve_types_num = len(set(value['curve_type'] for value in parameter.values()))

        self.new_rms = 0.0
        self.old_rms = None

        self.last_rms = 0
        self.count1 = 0
        self.count2 = 0
        self.count3 = 0

        self.t = 0
        self.t_limit = t_limit
        self.t_change = None
        self.first = True
        self.cum_reward = 0
        self.next_state = None
        self.epoch = -1
        self.action_valid_num = 0
        self.current_epoch_min_rms = 1
        self.exponent_new = 0
        self.exponent_old = 0

        self.min_state = copy.deepcopy(first_state)
        self.min_rms_init = True
        self.min_rms = None
        self.frame_count = 0

        self.data_target = {}
        for i in range(self.curve_types_num):
            self.data_target[i] = extract_target(target[i],self.curve_type[i])

        self.fitting_path = fitting

        self.train_model = train_model
        fig, self.ax = plt.subplots(1, 1, figsize=(16, 12))
        self.step_records = []  # Initialize the records list
        self.file_path = rf'{self.task_batch_path[0]}\results\records.txt'
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        # Clear the file content if it exists
        if os.path.exists(self.file_path):
            with open(self.file_path, 'w') as file:
                pass
        else:
            with open(self.file_path, 'w') as file:
                pass

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def update_parameters(self, action):
        action_valid = True
        step_size = (self.high - self.low) / self.step_size
        noise = random.gauss(mu=1.0, sigma=0.1)
        step_size = step_size * noise

        param_index = action // 2  # Determine which parameter to update
        action_type = action % 2  # Determine the type of action (increase, decrease, no change)

        if action_type == 0:
            direction = 1  # Increase
        elif action_type == 1:
            direction = -1  # Decrease
        else:
            direction = 0

        if self.state[param_index] >= self.high[param_index] or self.state[param_index] <= self.low[param_index]:
            flag = True
        else:
            flag = False
        self.state[param_index] += direction * step_size[param_index]

        # 判断是0.7+0.5>1还是第二种1+0.5>1:
        if self.state[param_index] >= self.high[param_index]:
            self.state[param_index] = self.high[param_index]
            if flag:
                action_valid = False
                self.action_valid_num += 1
        if self.state[param_index] <= self.low[param_index]:
            self.state[param_index] = self.low[param_index]
            if flag:
                action_valid = False
                self.action_valid_num += 1

        if direction != 0:
            # Find the parameter name corresponding to param_index
            param_replace = {}
            for name, param_info in self.parameter.items():
                if param_info['index'] == param_index:  # Find the matching parameter
                    param_replace[name] = self.state[param_index]

            try:
                # Read the model file content
                with open(self.model_path[0], 'r') as file:
                    content = file.readlines()  # Read all lines

                # Update parameter values in the content
                updated_content = []
                param_found = {key: False for key in param_replace.keys()}

                for line in content:
                    _data = line.replace('\n', '').split(' ')
                    index = 0
                    fitting_bool = False  # 修改参数值是否允许的信号
                    _param = ''
                    while index < len(_data):
                        if _data[index] in param_replace.keys():
                            fitting_bool = True
                            _param = _data[index]
                            param_found[_data[index]] = True
                        elif _data[index].strip() != '=' and fitting_bool:
                            fitting_bool = False
                            _data[index] = f'{param_replace[_param]}'
                        index += 1
                    updated_content.append(' '.join(_data) + '\n')

                with open(self.model_path[0], 'w') as file:
                    file.writelines(updated_content)

                missing_params = [param for param, found in param_found.items() if not found]
                if missing_params:
                    print(f"错误：以下参数未在模型文件中找到: {', '.join(missing_params)}")

            except FileNotFoundError:
                print(f"错误：模型文件 {self.model_path[0]} 未找到。")
            except Exception as e:
                print(f"错误：更新模型文件时发生异常 - {e}")

        return self.state, action_valid

    def step(self, action):
        reward = 0
        done1 = False
        done2 = False
        self.t += 1

        if self.first:
            self.first = False
            self.step_records.append({
                'state': copy.deepcopy(self.state),
                'Initial rms:': self.old_rms
            })

        print(f'rms:{self.old_rms}')
        updated_state, action_valid = self.update_parameters(action)

        for command in self.command:
            result = subprocess.run(command, shell=True, cwd=r'D:\Synopsys\Hspice_P-2019.06-SP1-1\WIN64',
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                print(f"command '{command}' Simulation failed, return code:", result.returncode)
            else:
                print(f"command '{command}' Simulation successful")

        data_fitting = {}
        for i in range(self.curve_types_num):
            data_fitting[i] = extract_reference(self.fitting_path[i])

        new_rms = {}
        for i in range(self.curve_types_num):
            new_rms[i] = calculate_rms(data_fitting[i], self.data_target[i])
        self.new_rms = np.mean(list(new_rms.values()))

        if self.new_rms < self.min_rms:
            self.min_rms = self.new_rms
            self.min_state = copy.deepcopy(updated_state)
            self.t_change = self.t - self.action_valid_num
            self.step_records.append({'Attention! RMS has reached a new minimum value.': copy.deepcopy(self.min_rms)})

        if self.new_rms == self.old_rms:
            reward = 0
        else:
            reward = np.around(1 / self.new_rms - 1 / self.old_rms)

        self.old_rms = self.new_rms
        self.cum_reward += reward
        self.step_records.append({
            'state': copy.deepcopy(self.state),
            'new_rms': self.new_rms,
            'action': action,
            'step': self.t,
            'reward': reward
        })

        if self.current_epoch_min_rms > self.new_rms:

            self.current_epoch_min_rms = self.new_rms

        if self.t >= self.t_limit:
            done2 = True

        if (done1 or done2) and self.train_model:
            with open(self.file_path, 'a') as f:
                for record in self.step_records:
                    f.write(f"{record}\n")
                f.write(f"Minimum RMSE obtained for all cycles: {self.min_rms}\n")
                f.write(f"Total rewards for this round: {self.cum_reward}\n")
                f.write(f"Current cycle:{self.epoch}\n")
                f.write(f"Minimum RMS value obtained in the current cycle: {self.current_epoch_min_rms}\n\n")

            print(self.new_rms)

        return self.state, float(reward), done1, done2, {'min_rms': self.min_rms,
                                                                        'change_t': self.t_change,
                                                                        'action_valid': action_valid,
                                                                        'new_rms': self.new_rms}

    def reset(self, **kwargs):
        if 'best_result' in kwargs:
            self.first_state = kwargs['best_result']
        else:
            # self.state = np.array(self.min_state[:])
            self.state = np.array(self.first_state[:])

        reset_modelcard(self.state, self.model_path[0], self.parameter)

        for command in self.command:
            result = subprocess.run(command, shell=True, cwd=r'D:\Synopsys\Hspice_P-2019.06-SP1-1\WIN64',
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                print(f"command '{command}' Simulation statement execution failed, return code:", result.returncode)
            else:
                print(f"command '{command}' Simulation statement execution successful.")

        data_fitting = {}
        for i in range(self.curve_types_num):
            data_fitting[i] = extract_reference(self.fitting_path[i])

        old_rms = {}
        for i in range(self.curve_types_num):
            old_rms[i] = calculate_rms(data_fitting[i], self.data_target[i])
        self.old_rms = np.mean(list(old_rms.values()))

        for i in range(self.curve_types_num):
            results_path = fr'{self.task_batch_path[0]}\results'
            destination_file = os.path.join(results_path, f'{os.path.basename(self.fitting_path[i])}_bestfitting.csv')
            print('2', destination_file)
            shutil.copy(self.fitting_path[i], destination_file)

        self.cum_reward = 0
        self.first = True
        self.t = 0
        self.t_change = 0
        self.new_rms = 0
        self.step_records = []
        self.next_state = None
        self.epoch += 1
        self.action_valid_num = 0
        self.current_epoch_min_rms = 100
        self.frame_count = 0

        if self.min_rms_init:
            self.min_rms = self.old_rms
            self.min_rms_init = False

        return self.state, {'initial_rms': self.old_rms, 'best_result': self.min_state, 'min_rms': self.min_rms}

    def render(self, mode='human', **args):
        # plt.ion()

        # data_fitting1 = extract_reference_cgc(self.fitting_path)
        #
        # fitting_color = 'black'
        # target_color = 'blue'
        #
        # self.ax.clear()
        # self.ax.plot(data_fitting1['vgs'], data_fitting1['cgc'], color=fitting_color, linestyle='-',
        #              label=f'Vbs={0.0}V', linewidth=2)
        # self.ax.plot(self.data_target1['vgs'], self.data_target1['cgc'], color=target_color, linestyle='--',
        #              label=f'Vbs={0.0}V', linewidth=2)
        # # self.ax.set_title(f'{self.t}_{self.state}')
        # self.ax.set_xlabel('Vgs', fontsize=20)
        # self.ax.set_ylabel('Cgc', fontsize=20)
        # self.ax.legend(fontsize=20)
        # self.ax.text(0.98, 0.02, '(a)', transform=self.ax.transAxes, fontsize=30, ha='right', va='bottom')
        #
        # plt.show()
        # plt.pause(0.02)


        # for i in range(self.curve_types_num):
        #     frame_directory = rf'{self.task_batch_path}\frames\{self.curve_type[i]}'
        #     if not os.path.exists(frame_directory):
        #         os.makedirs(frame_directory)

        #    frame_path = os.path.join(frame_directory, f'frames_{self.frame_count:04d}.png')
        #  plt.savefig(frame_path)

        for i in range(self.curve_types_num):
            data_bak= rf'{os.path.dirname(self.fitting_path[i])}\data_bak\{os.path.basename(self.fitting_path[i])}'
            if not os.path.exists(data_bak):
                os.makedirs(data_bak)

            destination_file = os.path.join(data_bak,f'{os.path.basename(self.fitting_path[i])}_{self.epoch}_{self.t}.csv')
            shutil.copy(self.fitting_path[i], destination_file)

        # self.frame_count += 1


