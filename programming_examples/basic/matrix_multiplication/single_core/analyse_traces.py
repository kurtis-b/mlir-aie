import os
import json
import re
import csv
import shutil

import matplotlib.pyplot as plt


def calculate_mmul_average(f):
    durations = []
    start_times = []
    end_times = []
    
    with open(f, 'r') as trace_file:
        for line in trace_file:
            events = json.loads(line)
            for event in events:
                if event['name'] == 'INSTR_EVENT_0' and event['ph'] == 'B':
                    start_times.append(event['ts'])
                if event['name'] == 'INSTR_EVENT_1' and event['ph'] == 'E':
                    end_times.append(event['ts'])
                if len(start_times) == 100 and len(end_times) == 100:
                    break
            if len(start_times) == 100 and len(end_times) == 100:
                break
    idx = 0
    for start in start_times:
        while start > end_times[idx]:
            if idx == len(end_times) - 1:
                break
            idx = idx + 1
        if idx == len(end_times) - 1:
            break
        durations.append(end_times[idx] - start)
    average_duration = sum(durations) / len(durations) if durations else 0
    return average_duration
  

def calculate_kernel_efficiency(f, kernel_dimensions, dtype_in):
    theoretical_performance_dict = { # MACs per cycle
        'i8': 256, 
        'i16': 64, 
        'bf16': 128
    }
    mmul_average = calculate_mmul_average(f)
    macs = int(kernel_dimensions[0]) * int(kernel_dimensions[1]) * int(kernel_dimensions[2])
    # print("mmul_avg: {} cycles, macs: {}, actual performance: {} MACs per cycle".format(mmul_average, macs, macs / mmul_average))
    theoretical_performance = theoretical_performance_dict[dtype_in]  
    return (macs / mmul_average) / theoretical_performance * 100


def load_json_files(directory):
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])
    data = []
    json_decoding_errors = []
    for file_path in json_files:
        if os.stat(os.path.join(directory, file_path)).st_size == 0:
            # print(f"Skipping empty file: {file}")
            continue
        file_name = os.path.splitext(file_path)[0]
        match = re.match(r'trace_mm_(\d+x\d+x\d+)_(\d+x\d+x\d+)_(\w+)_(\w+)', file_name)
        if match:
            if any('16' in dim for dim in match.group(2).split('x')):
                print(f"Skipping file with kernel {match.group(2)}: {file_path}")
                continue
            x_label = f"{match.group(2)}"
            try:
                data.append((x_label, calculate_kernel_efficiency(os.path.join(directory, file_path), match.group(2).split('x'), match.group(3))))
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {file_path}")
                json_decoding_errors.append(file_path)
                continue
    if json_decoding_errors:     
        print(f"JSON decoding errors occurred in the following files: {json_decoding_errors}")
    return data


def plot_data(data, plot_name, title):
    labels, values = zip(*sorted(data, key=lambda x: list(map(int, x[0].split('x')))))
    plt.figure()
    plt.bar(labels, values)
    plt.xlabel('Kernel Dimensions')
    plt.ylabel('Kernel Efficiency')
    plt.title(title, wrap=True)
    N = len(data)
    plt.xticks(rotation=45)  # Rotate x-axis labels and align them to the right
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2 # inch margin
    s = maxsize/plt.gcf().dpi*N+2*m
    margin = m/plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.tight_layout()
    plt.savefig(f'{plot_name}.pdf')
    plt.close()
    # plt.show()


def main():
    traces_dir = './traces'
    profiling_data_dir = './profiling_data'
    os.makedirs(profiling_data_dir, exist_ok=True)
    if not os.path.exists(traces_dir):
        print(f"Directory {traces_dir} does not exist.")
    else:
        for trace_file in sorted(os.listdir(traces_dir)):
            if trace_file.endswith('.json'):
                match = re.match(r'trace_mm_(\d+x\d+x\d+)_(\d+x\d+x\d+)_(\w+)_(\w+)\.json', trace_file)
                if match:
                    new_folder_name = f"traces_{match.group(1)}_{match.group(3)}_{match.group(4)}"
                    new_folder_path = os.path.join(profiling_data_dir, new_folder_name)
                    os.makedirs(new_folder_path, exist_ok=True)
                    shutil.copy(os.path.join(traces_dir, trace_file), new_folder_path)
                    os.remove(os.path.join(traces_dir, trace_file))
                    print(f"Copied {trace_file} to {new_folder_path}")

    empty_data_folders = []
    with open(os.path.join(profiling_data_dir, 'max_efficiency_data.csv'), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Workload Characteristics', 'Matrix Dimensions', 'Max Kernel Efficiency', 'Matrix Dimensions', 'Min Kernel Efficiency'])
        traces_to_plot = ['traces_64x1024x256', 'traces_128x1024x256', 'traces_256x1024x256']
        data_to_plot = {'i8_i8': [], 'i8_i16': [], 'bf16_bf16': [], 'bf16_f32': []}
        for folder in sorted(os.listdir(profiling_data_dir), reverse=True):
            if any(trace in folder for trace in traces_to_plot):
                print(f"Processing folder: {folder}")
                folder_path = os.path.join(profiling_data_dir, folder)
                match = re.match(r'traces_(\d+x\d+x\d+)_(\w+)_(\w+)', folder)
                if os.path.isdir(folder_path):
                    data = load_json_files(folder_path)
                    if data:
                        data_to_plot[f'{match.group(2)}_{match.group(3)}'].append((folder, data))
                        max_tuple = max(data, key=lambda x: x[1])
                        min_tuple = min(data, key=lambda x: x[1])
                        folder_name = folder.replace('traces_', '', 1)
                        csvwriter.writerow((folder_name, max_tuple[0], max_tuple[1], min_tuple[0], min_tuple[1]))
                    else:
                        empty_data_folders.append(folder)

        # Find all unique kernel dimensions across all data_to_plot values
        all_kernel_dimensions = set()
        for data in data_to_plot.values():
            for folder, data_list in data:
                for kernel_dim, _ in data_list:
                    all_kernel_dimensions.add(kernel_dim)
            print(all_kernel_dimensions)
            # Ensure each list in data_to_plot has all kernel dimensions, setting missing ones to 0
            for key, data in data_to_plot.items():
                for i, (folder, data_list) in enumerate(data):
                    existing_kernel_dims = {kernel_dim for kernel_dim, _ in data_list}
                    missing_kernel_dims = all_kernel_dimensions - existing_kernel_dims
                    print(existing_kernel_dims, missing_kernel_dims)
                    for missing_dim in missing_kernel_dims:
                        data_list.append((missing_dim, 0))
                    data[i] = (folder, sorted(data_list, key=lambda x: list(map(int, x[0].split('x')))))

        for data_pairs in data_to_plot.values():
            for folder, data in data_pairs:
                print(f"Plotting folder: {folder}")
                folder_path = os.path.join(profiling_data_dir, folder)
                match = re.match(r'traces_(\d+x\d+x\d+)_(\w+)_(\w+)', folder)
                if match:
                    plot_data(data, os.path.join(profiling_data_dir, folder), f'Kernel Efficiency for Application Workload: {match.group(1)}, Input Data Type: {match.group(2)}, Output Data Type: {match.group(3)}')

    if empty_data_folders:
        print(f"The following directories led to an empty data list: {empty_data_folders}")

if __name__ == "__main__":
    main()
