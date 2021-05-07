"""
Make all plots for paper
"""
import subprocess

if __name__ == "__main__":
    if 1 == 0:
        subprocess.run(['python', 'examples/proof_of_concept1.py'], check=True)
        subprocess.run(['python', 'examples/pccf_example.py'], check=True)
        subprocess.run(['python', 'examples/performance_pccf_signals.py'], check=True)
        subprocess.run(['python', 'proc_plot_temp_bars.py'], check=True)
        subprocess.run(['python', 'proc_plot_arl.py'], check=True)
        subprocess.run(['python', 'examples/performance_arls_simulation.py'], check=True)
    subprocess.run(['python', 'examples/proof_of_concept2.py'], check=True)
    subprocess.run(['python', 'examples/performance_detection_simulation.py'], check=True)
    # todo: fix warnings
    subprocess.run(['python', 'examples/performance_detection_signals.py'], check=True)
    subprocess.run(['python', 'proc_plot_temperature_signal.py'], check=True)
