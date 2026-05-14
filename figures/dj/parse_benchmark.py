import re
import csv
import sys

# Usage: python parse_benchmark.py results.txt output.csv
# Or pipe: cat results.txt | python parse_benchmark.py > output.csv

def parse_benchmark(text):
    rows = []

    # --- Serial ---
    serial_time = None
    m = re.search(r'Elapsed time\s*:\s*([\d.e+\-]+)\s*seconds', text)
    if m:
        serial_time = float(m.group(1))

    # --- Parallel blocks ---
    # Split by delta sections
    delta_blocks = re.split(r'={3,}\s*Parallel Delta-Stepping \(delta=(\d+)\)\s*={3,}', text)

    # delta_blocks[0] = serial section (already parsed)
    # delta_blocks[1] = delta value, delta_blocks[2] = content, ...
    i = 1
    while i < len(delta_blocks) - 1:
        delta = int(delta_blocks[i])
        block = delta_blocks[i + 1]
        i += 2

        # Split into per-thread blocks
        thread_sections = re.split(r'-{3,}\s*(\d+)\s*thread\(s\)\s*-{3,}', block)

        j = 1
        while j < len(thread_sections) - 1:
            threads = int(thread_sections[j])
            section = thread_sections[j + 1]
            j += 2

            row = {
                'type': 'parallel',
                'delta': delta,
                'threads': threads,
                'serial_time_s': serial_time,
            }

            def extract(pattern, text, cast=float, default=''):
                m = re.search(pattern, text)
                return cast(m.group(1)) if m else default

            row['preprocessing_s']        = extract(r'Preprocessing.*?:\s*([\d.e+\-]+)\s*seconds', section)
            row['sssp_time_s']             = extract(r'SSSP time\s*:\s*([\d.e+\-]+)\s*seconds', section)
            row['total_time_s']            = extract(r'Total\s*:\s*([\d.e+\-]+)\s*seconds', section)
            row['nodes_visited']           = extract(r'Nodes visited\s*:\s*(\d+)', section, int)
            row['parallel_relaxation_s']   = extract(r'\[timing\] parallel relaxation\s*:\s*([\d.e+\-]+)', section)
            row['parallel_merge_s']        = extract(r'\[timing\] parallel merge\s*:\s*([\d.e+\-]+)', section)
            row['find_min_bucket_s']       = extract(r'\[timing\] find min bucket\s*:\s*([\d.e+\-]+)', section)
            row['work_available_s']        = extract(r'\[timing\] work_available\s*:\s*([\d.e+\-]+)', section)
            row['snapshot_s']              = extract(r'\[timing\] snapshot\s*:\s*([\d.e+\-]+)', section)
            row['outgoing_clear_s']        = extract(r'\[timing\] outgoing clear\s*:\s*([\d.e+\-]+)', section)
            row['flatten_processed_s']     = extract(r'\[timing\] flatten processed\s*:\s*([\d.e+\-]+)', section)
            row['bucket_erase_s']          = extract(r'\[timing\] bucket erase\s*:\s*([\d.e+\-]+)', section)
            row['parallel_fraction_pct']   = extract(r'\[timing\] parallel fraction\s*:\s*([\d.e+\-]+)', section)
            row['effective_bandwidth_gbs'] = extract(r'Effective bandwidth\s*:\s*([\d.e+\-]+)', section)
            row['bandwidth_utilization_pct'] = extract(r'Bandwidth utilization:\s*([\d.e+\-]+)', section)
            row['effective_gflops']        = extract(r'Effective GFLOPS\s*:\s*([\d.e+\-]+)', section)

            # derived
            if row['sssp_time_s'] != '' and serial_time:
                row['sssp_speedup'] = round(serial_time / row['sssp_time_s'], 4)
            else:
                row['sssp_speedup'] = ''

            rows.append(row)

    # Add serial row
    if serial_time is not None:
        rows.insert(0, {
            'type': 'serial',
            'delta': '',
            'threads': 1,
            'serial_time_s': serial_time,
            'preprocessing_s': '',
            'sssp_time_s': serial_time,
            'total_time_s': serial_time,
            'nodes_visited': '',
            'parallel_relaxation_s': '',
            'parallel_merge_s': '',
            'find_min_bucket_s': '',
            'work_available_s': '',
            'snapshot_s': '',
            'outgoing_clear_s': '',
            'flatten_processed_s': '',
            'bucket_erase_s': '',
            'parallel_fraction_pct': '',
            'effective_bandwidth_gbs': '',
            'bandwidth_utilization_pct': '',
            'effective_gflops': '',
            'sssp_speedup': 1.0,
        })

    return rows

def main():
    if len(sys.argv) >= 2:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    rows = parse_benchmark(text)

    outfile = sys.argv[2] if len(sys.argv) >= 3 else 'benchmark_results.csv'

    fieldnames = [
        'type', 'delta', 'threads',
        'serial_time_s', 'preprocessing_s', 'sssp_time_s', 'total_time_s',
        'sssp_speedup', 'nodes_visited',
        'parallel_relaxation_s', 'parallel_merge_s', 'find_min_bucket_s',
        'work_available_s', 'snapshot_s', 'outgoing_clear_s',
        'flatten_processed_s', 'bucket_erase_s', 'parallel_fraction_pct',
        'effective_bandwidth_gbs', 'bandwidth_utilization_pct', 'effective_gflops',
    ]

    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {outfile}")

if __name__ == '__main__':
    main()