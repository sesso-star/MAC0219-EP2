#! /usr/bin/python3

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

######## Parsers ##########

def parse_results(results_dir, warp_range, size_range):
    if results_dir[-1] != '/':
        results_dir+='/'

    results = {}
    for nWarps in warp_range:
        results[nWarps] = {}

        for j in size_range:
            size = 2 ** j
            results[nWarps][size] = {}
            file_name = ""

            file_name = str(size) + "bible_" + str(nWarps) + "warps.log"
            result_file = open (results_dir + file_name)

            for line in result_file:
                match = re.search ('(\d+(\.|,)\d+)\s+seconds\s+time\s+elapsed\s+\(\s+\+\-\s+(\d+(\.|,)\d+)%\s+\)', line)
                if match:
                    avg = float (match.group(1).replace(",", "."))
                    std_dev = (float (match.group (3).replace(",", ".")) / 100) * avg
                    results[nWarps][size]["avg"] = avg
                    results[nWarps][size]["std_dev"] = std_dev

                match = re.search('Failed to launch rot13 kernel', line)
                if match:
                    results[nWarps][size]["avg"] = -0.007777
                    results[nWarps][size]["std_dev"] = 0
                    break
    return results

def parse_parallel_results(results_dir):
    return parse_results(results_dir, range(1, 33), range(0, 4))

def parse_sequential_results(results_dir):
    return parse_results(results_dir, range(1, 2), range(0, 4))                     

############## Plotters #################

class Plotter:
    def __init__(self, algorithm = "Encryptation algorithm", comment = ""):
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_subplot(111)
        self.algorithm = algorithm
        self.comment = comment
        self.reset_colors(12)
        self.init_data_vectors()
        self.legend_artist = None
    
    def reset(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_subplot(111)
        self.reset_colors(12)
        self.init_data_vectors()

    def reset_colors(self, nColors):
        self.cm = plt.get_cmap('gist_rainbow')
        self.ax.set_color_cycle([self.cm(1. * i / nColors) for i in range(nColors)])

    def init_data_vectors(self, threads_range=range(1, 33), size_range=range(0, 4)):
        self.warps = [i for i in threads_range]
        self.sizes = [2 ** i for i in size_range]

    def save(self, img_name=""):
        self.fig.savefig(img_name, dpi=200, bbox_extra_artists=(self.legend_artist,), bbox_inches='tight')
        self.reset()

    def show(self, img_name=""):
        self.save(img_name)
        # plt.show()
        self.reset()

    def get_timeXsize_lists(self, results):
        ylists = {nWarps: {"avg": [], "std_dev": []} for nWarps in self.warps}

        for size in self.sizes:
            for nWarps in self.warps:
                ylists[nWarps]["avg"].append(results[nWarps][size]["avg"])
                ylists[nWarps]["std_dev"].append(results[nWarps][size]["std_dev"])

        return ylists

    def get_timeXthread_lists(self, results):
        ylists = {size: {"avg": [], "std_dev": []} for size in self.sizes}

        for nWarps in self.warps:
            for size in self.sizes:
                ylists[size]["avg"].append(results[nWarps][size]["avg"])
                ylists[size]["std_dev"].append(results[nWarps][size]["std_dev"])
        return ylists

    def plot(self, xlist, ylist, marker="o", label=""):
        legend, = plt.plot(xlist, ylist["avg"], marker, mew=2, mfc='none', label=label, alpha=0.8)
        plt.plot(xlist, ylist["avg"], color='0.85', linewidth=0.5)
        self.ax.errorbar(xlist, ylist["avg"], yerr=ylist["std_dev"], ecolor='black', color='0.85', linewidth=0.5)
        return legend

    ##### Plot methods #####

    def timeXsize(self, results):
        self.reset_colors(32)

        ylists = self.get_timeXsize_lists(results)

        legend_handles = []
        legends = []

        for nWarps, ylist in ylists.items():
            th_legend = self.plot(self.sizes, ylist)

            legend_handles.append(th_legend)
            legends.append(str(nWarps) + " Warps")

        self.legend_artist = self.ax.legend(legend_handles, legends, bbox_to_anchor=(1.05, 1))


        plt.title('Time of execution X input size (' + self.algorithm + self.comment + ")")
        plt.ylabel('Time (s)')
        plt.xlabel('x Size of Bible (4452070 chars)')

        my_xticks = self.sizes
        plt.xticks(self.sizes, my_xticks)

        self.show("timeXsize_" + self.algorithm + self.comment)

    def timeXthread(self, results):
        self.reset_colors(8)

        ylists = self.get_timeXthread_lists(results)

        legend_handles = []
        legends = []

        for size, ylist in ylists.items():
            size_legend = self.plot(self.warps, ylist)

            legend_handles.append(size_legend)
            legends.append(str(size) + " x Bible")

        self.legend_artist = self.ax.legend(legend_handles, legends)

        plt.title('Time of execution X number of threads (' + self.algorithm + self.comment + ")")
        plt.ylabel('Time (s)')
        plt.xlabel('x 32 threads')

        my_xticks = self.warps
        plt.xticks(self.warps, my_xticks)

        self.show("timeXthread_" + self.algorithm + self.comment)


    def compare_timeXsize(self, results, range1, results2, range2, nColors=10, group1="group 1", group2="group 2"):
        label1 = group1
        label2 = group2

        self.init_data_vectors(threads_range=range1)
        ylists = self.get_timeXsize_lists(results)
        self.reset_colors(3)

        legend_handles = []
        legends = []

        for nWarps, ylist in ylists.items():
            th_legend = self.plot(self.sizes, ylist)

            legend_handles.append(th_legend)
            legends.append("parallel (" + str(nWarps) + " x 32 threads)")

        self.init_data_vectors(threads_range=range2)
        ylists2 = self.get_timeXsize_lists(results2)
        for nWarps, ylist in ylists2.items():
            th_legend = self.plot(self.sizes, ylist, marker="s")

            legend_handles.append(th_legend)
            legends.append("sequential")

        ### Legends ###
        self.legend_artist = self.ax.legend(legend_handles, legends)

        plt.title('Comparision of time of execution X size of input (' + self.algorithm + " " + self.comment + ")")
        plt.ylabel('Time (s)')
        plt.xlabel('x Size of Bible (4452070 chars)')

        my_xticks = self.sizes
        plt.xticks(self.sizes, my_xticks)

        self.show("compare_timeXsize_" + self.algorithm + self.comment)


if __name__ == '__main__':
    argv = sys.argv
    plot = Plotter()
    plot.algorithm = "Rot-13"

    plot.comment = " - Parallel"
    parallel_results = parse_parallel_results(sys.argv[1])
    sequential_results = parse_sequential_results(argv[2])

    plot.timeXsize(parallel_results)
    plot.timeXthread(parallel_results)

    plot.comment = " - Sequential"
    plot.init_data_vectors(threads_range=range(1, 2))
    plot.timeXsize(sequential_results)

    plot.comment = " - Parallel vs Sequential"
    plot.compare_timeXsize(parallel_results, range(32,33), sequential_results, range(1, 2), group1="Parallel", group2="Sequential")
