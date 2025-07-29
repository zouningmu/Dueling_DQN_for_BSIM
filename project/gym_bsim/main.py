import pandas as pd
import argparse
import time
import DQN_training as tm
import traceback


def task_list(task_batch_name,curve_types):
    df = pd.read_excel(rf'D:\BSIM\test_data\task_list.xls')
    df = df[df['task_batch_name'] == task_batch_name]
    df_res = df[df['curve_type'].isin(curve_types)]
    task_dick = df_res.to_dict()
    print(task_dick)

    return task_dick


def task_record(task_record_file, task_dick):
    with open(task_record_file, 'w') as file:
        file.write(','.join(task_dick.keys()) + '\n')
        print(task_dick.keys())
        file.write(','.join(task_dick.values()))
        print(task_dick.values())


def param_list(curve_types):
    df = pd.read_excel(r'D:\BSIM\test_data\param_list.xls')
    merged_dict = {}

    for curve_type in curve_types:
        param_data = df[df['curve_type'] == curve_type]
        param_data.set_index('param_name', inplace=True)
        param_dict = param_data.to_dict(orient='index')
        merged_dict.update(param_dict)

    for idx, key in enumerate(merged_dict):
        # 参数个数根据最大index设置
        # 一定要满足max(index) == 实际需要拟合的参数个数
        merged_dict[key]['index'] = idx

    # print(merged_dict)
    return merged_dict


def main(param_dict, curve_types, task_dick):
    parser = argparse.ArgumentParser(description='DQN Training Script')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Total number of training episodes')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units in the neural network')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor for rewards')
    parser.add_argument('--target_update', type=int, default=10, help='Frequency of target network updates')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (e.g., "cpu" or "cuda")')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Size of the replay buffer')
    parser.add_argument('--minimal_size', type=int, default=256, help='Minimum size of the replay buffer before training starts')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epsilon_initial', type=float, default=0.7, help='Initial value of epsilon for epsilon-greedy policy')
    parser.add_argument('--epsilon_final', type=float, default=0.01, help='Final value of epsilon for epsilon-greedy policy')
    parser.add_argument('--epsilon', type=float, default=0.8, help='Epsilon value for choosing whether to execute past optimal steps')
    parser.add_argument('--t_limit', type=int, default=10, help='Number of actions per episode')
    parser.add_argument('--step_size', type=int, default=5, help='Step size for parameter adjustments')
    parser.add_argument('--task_batch_path', type=dict, default=task_dick['task_batch_path'], help='Path to task path')
    parser.add_argument('--curve_type', type=dict, default=task_dick['curve_type'], help='Path to curve type')
    parser.add_argument('--target_path', type=dict, default=task_dick['target_path'], help='Path to target data')
    parser.add_argument('--fitting_path', type=dict, default=task_dick['fitting_path'], help='Path to fitting data')
    parser.add_argument('--model_path', type=str, default=task_dick['model_path'], help='Path to model file')
    parser.add_argument('--sp_path', type=dict, default=task_dick['sp_path'], help='Path1 to SP file')
    parser.add_argument('--parameter', type=dict, default=param_dict, help='Parameter dictionary')
    args = parser.parse_args()
    for i in range(5):
        best_result = tm.train(args)
        for idx, param_name in enumerate(args.parameter):
            args.parameter[param_name]['initial'] = best_result[idx]
            args.parameter[param_name]['high'] += best_result[idx]
            args.parameter[param_name]['high'] /= 2
            args.parameter[param_name]['low'] += best_result[idx]
            args.parameter[param_name]['low'] /= 2


if __name__ == "__main__":
    task_batch_name = 'your_task_batch'
    curve_types = ['cgg_vgs']
    task_dick = task_list(task_batch_name, curve_types)
    # print('task_dick', task_dick)

    param_dict = param_list(curve_types)
    start_time = time.time()
    train_state = 'Sucess'
    try:
        main(param_dict, curve_types, task_dick)
    except Exception as e:
        train_state = 'Fail'
        print(f"Error occurred: {e}")
        traceback.print_exc()

    end_time = time.time()